import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import threading
from threading import Thread
from queue import Queue
from typing import List, Tuple, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from roop.face_processor import find_faces, swap_face, get_face_reference
from roop.utilities import get_device_info

# GPU optimization settings
BATCH_SIZE = 8
PREFETCH_SIZE = 16
USE_MIXED_PRECISION = True  # Use FP16 precision on capable GPUs
MAX_THREADS = os.cpu_count()
FRAME_QUEUE_SIZE = 32
RESULT_QUEUE_SIZE = 32

# Locks for thread safety
GPU_LOCK = threading.Lock()
QUEUE_LOCK = threading.Lock()

# CUDA streams for parallel processing
CUDA_STREAMS = []

def initialize_cuda_streams(num_streams=2):
    """Initialize CUDA streams for parallel processing"""
    global CUDA_STREAMS
    if torch.cuda.is_available():
        CUDA_STREAMS = [torch.cuda.Stream() for _ in range(num_streams)]

def preprocess_frame(frame: np.ndarray) -> torch.Tensor:
    """Preprocess frame for GPU processing"""
    if frame is None:
        return None
    
    # Convert to float32 and normalize
    tensor = torch.from_numpy(frame).float().div(255.0)
    # Add batch dimension and convert to channel-first format (BCHW)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        tensor = tensor.cuda(non_blocking=True)
    
    return tensor

def postprocess_frame(tensor: torch.Tensor) -> np.ndarray:
    """Convert processed tensor back to numpy array"""
    if tensor is None:
        return None
    
    # Move to CPU if on GPU
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # Convert to numpy array
    frame = tensor.squeeze(0).permute(1, 2, 0).mul(255).byte().numpy()
    return frame

def batch_process_frames(frames: List[np.ndarray], process_func: callable) -> List[np.ndarray]:
    """Process frames in batches for better GPU utilization"""
    if not frames:
        return []
    
    batch_size = min(BATCH_SIZE, len(frames))
    results = []
    
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i+batch_size]
        
        # Preprocess batch for GPU
        if torch.cuda.is_available():
            with torch.cuda.stream(CUDA_STREAMS[0]):
                tensors = [preprocess_frame(frame) for frame in batch]
                
                # Process with mixed precision if enabled
                if USE_MIXED_PRECISION:
                    with autocast():
                        processed_tensors = [process_func(tensor) for tensor in tensors]
                else:
                    processed_tensors = [process_func(tensor) for tensor in tensors]
                
                # Convert back to numpy
                processed_frames = [postprocess_frame(tensor) for tensor in processed_tensors]
                results.extend(processed_frames)
        else:
            # CPU fallback
            for frame in batch:
                processed = process_func(frame)
                results.append(processed)
    
    # Synchronize CUDA streams
    if torch.cuda.is_available():
        for stream in CUDA_STREAMS:
            stream.synchronize()
    
    return results

class FrameProcessor:
    def __init__(self, source_face: np.ndarray, face_positions: List[int] = None):
        self.source_face = source_face
        self.face_positions = face_positions
        self.input_queue = Queue(maxsize=FRAME_QUEUE_SIZE)
        self.output_queue = Queue(maxsize=RESULT_QUEUE_SIZE)
        self.stop_event = threading.Event()
        self.worker_threads = []
        self.total_frames = 0
        self.processed_frames = 0
        
        # Initialize CUDA streams
        initialize_cuda_streams()
    
    def start_workers(self, num_workers: int = None):
        """Start worker threads for parallel processing"""
        if num_workers is None:
            # Determine optimal number of workers based on hardware
            if torch.cuda.is_available():
                num_workers = min(4, torch.cuda.device_count() * 2)  # 2 workers per GPU
            else:
                num_workers = min(MAX_THREADS // 2, 4)  # Use half of available CPU cores
        
        for _ in range(num_workers):
            thread = Thread(target=self._process_frames_worker)
            thread.daemon = True
            thread.start()
            self.worker_threads.append(thread)
    
    def _process_frames_worker(self):
        """Worker thread for processing frames"""
        while not self.stop_event.is_set():
            try:
                # Get a frame from the input queue
                idx, frame = self.input_queue.get(timeout=1.0)
                if frame is None:
                    self.input_queue.task_done()
                    continue
                
                # Process the frame
                processed_frame = self._process_single_frame(frame)
                
                # Put the processed frame in the output queue
                self.output_queue.put((idx, processed_frame))
                
                # Update progress
                with QUEUE_LOCK:
                    self.processed_frames += 1
                
                self.input_queue.task_done()
            except Exception as e:
                print(f"Error in frame processing worker: {e}")
    
    def _process_single_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame with face swapping"""
        if frame is None:
            return None
        
        # Find faces in the frame
        faces = find_faces(frame)
        if not faces:
            return frame
        
        # Filter faces based on position if specified
        if self.face_positions:
            selected_faces = []
            for position in self.face_positions:
                if position < len(faces):
                    selected_faces.append(faces[position])
            faces = selected_faces
        
        # Apply face swap to each face
        result = frame.copy()
        for face in faces:
            with GPU_LOCK:  # Ensure GPU operations are thread-safe
                result = swap_face(self.source_face, face, result)
        
        return result
    
    def add_frame(self, idx: int, frame: np.ndarray):
        """Add a frame to the processing queue"""
        self.input_queue.put((idx, frame))
        with QUEUE_LOCK:
            self.total_frames += 1
    
    def get_processed_frame(self, timeout: float = 5.0) -> Tuple[int, np.ndarray]:
        """Get a processed frame from the output queue"""
        return self.output_queue.get(timeout=timeout)
    
    def frame_done(self):
        """Mark a frame as done"""
        self.output_queue.task_done()
    
    def stop(self):
        """Stop all worker threads"""
        self.stop_event.set()
        for thread in self.worker_threads:
            thread.join(timeout=2.0)
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def process_video(source_face: np.ndarray, target_path: str, output_path: str, 
                 temp_frame_dir: str = None, face_positions: List[int] = None) -> bool:
    """Process a video with optimized GPU utilization"""
    try:
        # Open the target video
        video_capture = cv2.VideoCapture(target_path)
        if not video_capture.isOpened():
            print("Error: Could not open video.")
            return False
        
        # Get video properties
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Create frame processor
        processor = FrameProcessor(source_face, face_positions)
        processor.start_workers()
        
        # Create a thread for reading frames
        frame_reader_stop = threading.Event()
        frames_read = 0
        
        def frame_reader():
            nonlocal frames_read
            idx = 0
            while not frame_reader_stop.is_set():
                ret, frame = video_capture.read()
                if not ret:
                    break
                
                # Add frame to processing queue
                processor.add_frame(idx, frame)
                frames_read += 1
                idx += 1
        
        # Start frame reader thread
        reader_thread = Thread(target=frame_reader)
        reader_thread.daemon = True
        reader_thread.start()
        
        # Process and write frames
        results = {}
        next_frame_to_write = 0
        
        while next_frame_to_write < total_frames:
            try:
                # Get processed frame
                idx, processed_frame = processor.get_processed_frame()
                results[idx] = processed_frame
                processor.frame_done()
                
                # Write frames in order
                while next_frame_to_write in results:
                    output_video.write(results[next_frame_to_write])
                    del results[next_frame_to_write]
                    next_frame_to_write += 1
                    
                # Print progress
                if next_frame_to_write % 10 == 0:
                    progress = (next_frame_to_write / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({next_frame_to_write}/{total_frames})")
            
            except Exception as e:
                print(f"Error processing frame: {e}")
                # If we've read all frames but some processing failed, increment to continue
                if frames_read >= total_frames and next_frame_to_write < total_frames:
                    next_frame_to_write += 1
        
        # Stop processing
        frame_reader_stop.set()
        reader_thread.join()
        processor.stop()
        
        # Release resources
        video_capture.release()
        output_video.release()
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True
    
    except Exception as e:
        print(f"Error processing video: {e}")
        return False

def get_device_stats():
    """Get GPU/CPU device statistics"""
    stats = {}
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        stats['gpu_count'] = device_count
        stats['devices'] = []
        
        for i in range(device_count):
            device_stats = {
                'name': torch.cuda.get_device_name(i),
                'memory_allocated': torch.cuda.memory_allocated(i) / (1024**2),  # MB
                'memory_reserved': torch.cuda.memory_reserved(i) / (1024**2),  # MB
                'max_memory_allocated': torch.cuda.max_memory_allocated(i) / (1024**2)  # MB
            }
            stats['devices'].append(device_stats)
    else:
        stats['gpu_count'] = 0
        stats['cpu_count'] = os.cpu_count()
    
    return stats

def optimize_batch_size():
    """Dynamically adjust batch size based on GPU memory"""
    global BATCH_SIZE
    
    if not torch.cuda.is_available():
        BATCH_SIZE = 1
        return
    
    # Get GPU memory info
    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory
    total_memory_gb = total_memory / (1024**3)  # Convert to GB
    
    # Adjust batch size based on available memory
    if total_memory_gb >= 20:  # High-end GPU (>= 20GB)
        BATCH_SIZE = 16
    elif total_memory_gb >= 12:  # Mid-range GPU (>= 12GB)
        BATCH_SIZE = 8
    elif total_memory_gb >= 6:  # Low-end GPU (>= 6GB)
        BATCH_SIZE = 4
    else:  # Very low memory
        BATCH_SIZE = 2
    
    print(f"Optimized batch size: {BATCH_SIZE} (GPU memory: {total_memory_gb:.1f} GB)")

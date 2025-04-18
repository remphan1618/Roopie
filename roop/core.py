import cv2
import threading
import numpy as np
import onnxruntime
import torch
import time
from typing import Any, List, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import roop.globals
import roop.processors.frame.core as frame_processors
from roop.face_analyser import get_one_face, get_many_faces
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

# GPU optimization settings
MAX_BATCH_SIZE = 16  # Adjust based on GPU VRAM
MAX_WORKERS = 8      # Worker threads for parallel processing
GPU_MEMORY_FRACTION = 0.9  # Use more GPU memory

# Global variables for model caching
MODEL_CACHE = {}
MODEL_CACHE_LOCK = threading.Lock()

def get_options():
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    if torch.cuda.is_available():
        options.intra_op_num_threads = torch.cuda.device_count() if torch.cuda.device_count() > 0 else 1
        # Add memory optimization
        options.enable_mem_pattern = True
        options.enable_mem_reuse = True
    return options

def get_providers():
    providers = []
    if torch.cuda.is_available():
        providers.append('CUDAExecutionProvider')
    providers.append('CPUExecutionProvider')
    return providers

def pre_process():
    if torch.cuda.is_available():
        # Optimize CUDA for better performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        # Allow TensorFloat32 for faster computation on Ampere GPUs
        torch.set_float32_matmul_precision('high')
        # Set memory fraction to use
        for device_idx in range(torch.cuda.device_count()):
            torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, device_idx)

def batch_process_frames(frames: List[np.ndarray], process_func: Callable) -> List[np.ndarray]:
    """Process frames in batches for better GPU utilization"""
    if not frames:
        return []
    
    start_time = time.time()
    results = []
    batch_size = min(MAX_BATCH_SIZE, len(frames))
    
    # Process in batches
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]
        
        # Convert batch to tensor for GPU processing if using PyTorch
        if torch.cuda.is_available():
            batch_tensors = [torch.from_numpy(frame).cuda() for frame in batch]
            # Process each frame in the batch
            processed_batch = []
            for j, frame_tensor in enumerate(batch_tensors):
                # Convert tensor back to numpy for processing
                processed = process_func(frame_tensor.cpu().numpy() if frame_tensor.is_cuda else frame_tensor.numpy())
                processed_batch.append(processed)
            results.extend(processed_batch)
        else:
            # CPU fallback - still use batching for organization
            for frame in batch:
                processed = process_func(frame)
                results.append(processed)
    
    fps = len(frames) / (time.time() - start_time)
    print(f"Processed {len(frames)} frames at {fps:.2f} FPS")
    
    return results

def parallel_process_frames(frames: List[np.ndarray], process_func: Callable) -> List[np.ndarray]:
    """Process frames in parallel using thread pool"""
    results = [None] * len(frames)
    
    def process_frame(idx: int, frame: np.ndarray):
        results[idx] = process_func(frame)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_frame, i, frame) for i, frame in enumerate(frames)]
        # Wait for all futures to complete
        for future in as_completed(futures):
            # Handle any exceptions
            if future.exception():
                print(f"Error processing frame: {future.exception()}")
    
    return results

def load_model(model_path: str):
    """Load model with caching for better performance"""
    with MODEL_CACHE_LOCK:
        if model_path not in MODEL_CACHE:
            options = get_options()
            providers = get_providers()
            MODEL_CACHE[model_path] = onnxruntime.InferenceSession(model_path, providers=providers, sess_options=options)
        return MODEL_CACHE[model_path]

def create_optimization_info(model_path: str):
    """Create optimization info for ONNX model"""
    if not torch.cuda.is_available():
        return
        
    try:
        import onnxruntime.transformers.optimizer as optimizer
        
        opt_model_path = model_path.replace('.onnx', '_optimized.onnx')
        
        # Only optimize if doesn't exist
        if not os.path.exists(opt_model_path):
            # Basic optimization settings
            optimization_options = optimizer.OptimizationConfig(
                optimization_level=99,  # Maximum optimization
                enable_gelu_approximation=True,
                enable_layer_norm_optimization=True,
                enable_attention_optimization=True,
                use_gpu=True
            )
            
            # Optimize model
            optimizer.optimize_model(
                model_path,
                opt_model_path,
                optimization_options=optimization_options
            )
            
            return opt_model_path
    except ImportError:
        print("ONNX Runtime Transformers not available, skipping optimization")
        return model_path
    except Exception as e:
        print(f"Error optimizing model: {e}")
        return model_path

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def clean_up():
    """Clean up resources and release memory"""
    # Clear model cache
    with MODEL_CACHE_LOCK:
        MODEL_CACHE.clear()
    
    # Release GPU memory
    clear_gpu_memory()

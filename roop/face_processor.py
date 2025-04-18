import os
import insightface
import cv2
import numpy as np
import threading
import torch
from typing import List, Dict, Tuple, Optional, Union, Any

from modules.globals import Global
from modules.face import Face

# Define model paths
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

# Class to manage model loading as singletons
class ModelLoader:
    _models = {}
    _lock = threading.Lock()
    
    @classmethod
    def get_inswapper_model(cls):
        with cls._lock:
            key = "inswapper"
            if key not in cls._models:
                model_path = os.path.join(MODELS_DIR, "inswapper_128.onnx")
                device_id = 0  # Use first GPU
                model = insightface.model_zoo.get_model(model_path, providers=Global.execution_providers)
                cls._models[key] = model
            return cls._models[key]

# Main face swapper class
class FaceProcessor:
    def __init__(self):
        self.inswapper = None
        self.face_cache = {}
        self.cache_lock = threading.Lock()
        self.device = 'cuda' if Global.cuda_available else 'cpu'
        self.face_debugger = None
    
    def _load_models(self):
        """Lazy-load models when needed"""
        if self.inswapper is None:
            self.inswapper = ModelLoader.get_inswapper_model()
    
    def _preprocess_source_face(self, source_face: Face, enhance_face: bool = False) -> Face:
        """Preprocess source face for swapping"""
        # If enhancement is needed, apply it here
        return source_face
    
    def process_frame(self, source_face: Face, target_frame: np.ndarray, target_faces: List[Face], 
                      enhance_face: bool = False, swap_all: bool = False, 
                      face_mask_offsets: Tuple[int, int, int, int] = (0, 0, 0, 0)) -> np.ndarray:
        """
        Process a frame to swap faces.
        
        Args:
            source_face: Face to use as source
            target_frame: Frame to modify
            target_faces: List of faces detected in the frame
            enhance_face: Whether to enhance the source face
            swap_all: Whether to swap all faces in the frame
            face_mask_offsets: Offsets for face mask (top, right, bottom, left)
            
        Returns:
            Modified frame
        """
        self._load_models()
        
        # Make a copy of the target frame to avoid modifying the original
        result_frame = target_frame.copy()
        
        # Preprocess the source face if needed
        processed_source_face = self._preprocess_source_face(source_face, enhance_face)
        
        # If no target faces, return the original frame
        if not target_faces:
            return result_frame
            
        # Determine which faces to swap
        faces_to_swap = target_faces if swap_all else [target_faces[0]]
        
        # Perform face swapping with FP16 optimization if enabled
        with torch.cuda.amp.autocast(enabled=Global.fp16_enabled):
            for target_face in faces_to_swap:
                # Calculate the face hash for caching
                face_hash = hash((source_face.embedding.tobytes(), target_face.embedding.tobytes()))
                
                # Use cached result if available
                with self.cache_lock:
                    if face_hash in self.face_cache:
                        result_frame = self.face_cache[face_hash]
                        continue
                
                # Perform actual face swapping
                try:
                    # Convert to proper format for inswapper if needed
                    if len(result_frame.shape) == 2:
                        swap_frame = cv2.cvtColor(result_frame, cv2.COLOR_GRAY2BGR)
                    elif result_frame.shape[2] == 4:
                        swap_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGRA2BGR)
                    else:
                        swap_frame = result_frame
                    
                    # Use CUDA for tensor operations if available
                    if Global.cuda_available:
                        with torch.no_grad():
                            # Convert tensors to GPU
                            result_frame = self.inswapper.get(
                                swap_frame, 
                                target_face.kps, 
                                processed_source_face.embedding, 
                                paste_back=True, 
                                face_mask_offsets=face_mask_offsets
                            )
                    else:
                        result_frame = self.inswapper.get(
                            swap_frame, 
                            target_face.kps, 
                            processed_source_face.embedding, 
                            paste_back=True,
                            face_mask_offsets=face_mask_offsets
                        )
                    
                    # Cache the result for future use
                    with self.cache_lock:
                        if len(self.face_cache) > 100:  # Limit cache size
                            self.face_cache.clear()
                        self.face_cache[face_hash] = result_frame.copy()
                        
                except Exception as e:
                    print(f"Error during face swap: {e}")
                    continue
                
        return result_frame
        
    def swap_face_with_mask(self, source_face: Face, target_frame: np.ndarray, target_face: Face,
                            mask: np.ndarray, enhance_face: bool = False, 
                            face_mask_offsets: Tuple[int, int, int, int] = (0, 0, 0, 0)) -> np.ndarray:
        """
        Swap face with custom mask instead of default face mask
        
        Args:
            source_face: Face to use as source
            target_frame: Frame to modify
            target_face: Target face to replace
            mask: Custom mask to use
            enhance_face: Whether to enhance the source face
            face_mask_offsets: Offsets for face mask (top, right, bottom, left)
            
        Returns:
            Modified frame
        """
        self._load_models()
        
        # Make a copy of the target frame to avoid modifying the original
        result_frame = target_frame.copy()
        
        # Preprocess the source face if needed
        processed_source_face = self._preprocess_source_face(source_face, enhance_face)
        
        # Perform face swapping with custom mask
        try:
            with torch.cuda.amp.autocast(enabled=Global.fp16_enabled):
                # Convert to proper format for inswapper if needed
                if len(result_frame.shape) == 2:
                    swap_frame = cv2.cvtColor(result_frame, cv2.COLOR_GRAY2BGR)
                elif result_frame.shape[2] == 4:
                    swap_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGRA2BGR)
                else:
                    swap_frame = result_frame
                
                # Use custom mask for face swapping
                swapped_face = self.inswapper.get(
                    swap_frame,
                    target_face.kps,
                    processed_source_face.embedding,
                    paste_back=False,
                    face_mask_offsets=face_mask_offsets
                )
                
                # Apply the custom mask
                if mask is not None:
                    # Ensure mask is properly sized
                    if mask.shape[:2] != swapped_face.shape[:2]:
                        mask = cv2.resize(mask, (swapped_face.shape[1], swapped_face.shape[0]))
                        
                    # Convert mask to 3-channel if it's single channel
                    if len(mask.shape) == 2:
                        mask = mask[:, :, np.newaxis]
                        mask = np.repeat(mask, 3, axis=2)
                    
                    # Apply the mask
                    result_frame = swapped_face * (mask / 255.0) + swap_frame * (1 - mask / 255.0)
                    result_frame = result_frame.astype(np.uint8)
                else:
                    # Use default inswapper behavior if no mask provided
                    result_frame = self.inswapper.get(
                        swap_frame,
                        target_face.kps,
                        processed_source_face.embedding,
                        paste_back=True,
                        face_mask_offsets=face_mask_offsets
                    )
                
        except Exception as e:
            print(f"Error during face swap with mask: {e}")
        
        return result_frame
    
    def batch_process(self, source_face: Face, target_frames: List[np.ndarray], 
                      target_faces_per_frame: List[List[Face]], enhance_face: bool = False, 
                      swap_all: bool = False, batch_size: int = 8) -> List[np.ndarray]:
        """
        Process multiple frames in batches for better performance
        
        Args:
            source_face: Face to use as source
            target_frames: List of frames to modify
            target_faces_per_frame: List of detected faces for each frame
            enhance_face: Whether to enhance the source face
            swap_all: Whether to swap all faces in each frame
            batch_size: Number of frames to process at once
            
        Returns:
            List of modified frames
        """
        result_frames = []
        
        # Process frames in batches
        for i in range(0, len(target_frames), batch_size):
            batch_frames = target_frames[i:i+batch_size]
            batch_faces = target_faces_per_frame[i:i+batch_size]
            
            # Process each frame in the batch
            # This could be optimized further with batch processing if inswapper supports it
            batch_results = []
            for frame, faces in zip(batch_frames, batch_faces):
                processed_frame = self.process_frame(source_face, frame, faces, 
                                                    enhance_face, swap_all)
                batch_results.append(processed_frame)
                
            result_frames.extend(batch_results)
        
        return result_frames

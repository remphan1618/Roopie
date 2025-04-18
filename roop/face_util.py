import cv2
import numpy as np
import os
import torch
from typing import Any, List, Callable, Optional, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from roop.utilities import resolve_relative_path, conditional_download

import insightface
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from insightface.utils import face_align

# Global variables for models to avoid reloading
FACE_ANALYSER = None
FACE_ANALYSER_LOCK = Lock()
THREAD_LOCK = Lock()
FACE_DETECT_MODEL = None
FACE_SWAP_MODEL = None
DEVICE = None
PROVIDERS = None
EXECUTION_PROVIDER = None
BATCH_SIZE = 8  # Default batch size for GPU processing

MAX_WORKERS = max(1, os.cpu_count() // 2)
THREAD_POOL = ThreadPoolExecutor(max_workers=MAX_WORKERS)

def initialize(execution_providers: List[str]) -> List[str]:
    global FACE_ANALYSER, FACE_DETECT_MODEL, FACE_SWAP_MODEL, DEVICE, PROVIDERS, EXECUTION_PROVIDER
    
    # Initialize GPU device
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
        # Use larger fraction of GPU memory
        torch.cuda.set_per_process_memory_fraction(0.9)
        # Optimize CUDA memory allocation
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        DEVICE = torch.device('cpu')
    
    PROVIDERS = execution_providers
    for execution_provider in execution_providers:
        if execution_provider == 'CUDAExecutionProvider':
            EXECUTION_PROVIDER = execution_provider
            break
        if execution_provider == 'CPUExecutionProvider':
            EXECUTION_PROVIDER = execution_provider
    
    return PROVIDERS

def get_face_analyser():
    global FACE_ANALYSER
    
    # Use lock to prevent multiple initializations in threaded environments
    with FACE_ANALYSER_LOCK:
        if FACE_ANALYSER is None:
            FACE_ANALYSER = FaceAnalysis(
                name='buffalo_l',
                root=resolve_relative_path('../models'),
                providers=[EXECUTION_PROVIDER] if EXECUTION_PROVIDER else PROVIDERS
            )
            # Use a lower detection threshold for better face detection
            FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.25)
    
    return FACE_ANALYSER

def get_face_swap_model():
    global FACE_SWAP_MODEL
    
    if FACE_SWAP_MODEL is None:
        model_path = resolve_relative_path('../models/inswapper_128.onnx')
        # Download model if needed
        conditional_download(
            model_path,
            'https://huggingface.co/datasets/levihsu/ERNet/resolve/main/inswapper_128.onnx'
        )
        # Load model with GPU optimization settings
        FACE_SWAP_MODEL = insightface.model_zoo.get_model(model_path, providers=[EXECUTION_PROVIDER] if EXECUTION_PROVIDER else PROVIDERS)
    
    return FACE_SWAP_MODEL

def find_faces(image: np.ndarray) -> List[Face]:
    try:
        return get_face_analyser().get(image)
    except Exception as e:
        print(f"Error in face detection: {e}")
        return []

def get_face_single(image: np.ndarray, position: int = 0) -> Optional[Face]:
    faces = find_faces(image)
    if position >= len(faces):
        return None
    return faces[position]

def prepare_batch(source_faces: List[np.ndarray], target_faces: List[np.ndarray]) -> Tuple[List, List, int]:
    batch_source = []
    batch_target = []
    batch_size = min(BATCH_SIZE, len(source_faces), len(target_faces))
    
    for i in range(batch_size):
        batch_source.append(source_faces[i])
        batch_target.append(target_faces[i])
    
    return batch_source, batch_target, batch_size

def batch_face_swap(source_faces: List[np.ndarray], target_faces: List[np.ndarray], target_image: np.ndarray) -> np.ndarray:
    """Process face swaps in batches for better GPU utilization"""
    if not source_faces or not target_faces:
        return target_image
    
    result_image = target_image.copy()
    model = get_face_swap_model()
    
    # Process in batches
    for i in range(0, len(source_faces), BATCH_SIZE):
        batch_source = source_faces[i:i+BATCH_SIZE]
        batch_target = target_faces[i:i+BATCH_SIZE]
        
        # Move batches to GPU
        if torch.cuda.is_available():
            batch_tensors = [torch.from_numpy(face).to(DEVICE) for face in batch_source]
            # Process batched face swaps
            for j in range(len(batch_source)):
                result_image = model.get(result_image, batch_target[j], batch_source[j])
        else:
            # Fallback to sequential processing
            for j in range(len(batch_source)):
                result_image = model.get(result_image, batch_target[j], batch_source[j])
    
    return result_image

def swap_face(source_face: np.ndarray, target_face: Face, target_image: np.ndarray) -> np.ndarray:
    model = get_face_swap_model()
    return model.get(target_image, target_face, source_face)

def pre_check() -> bool:
    download_path = resolve_relative_path('../models/inswapper_128.onnx')
    return os.path.isfile(download_path)

def extract_face(image: np.ndarray, face: Face, crop_size: int = 128) -> np.ndarray:
    # Create a copy of the image array to prevent modifying the original
    image = image.copy()
    
    # Use insightface's face_align for better alignment
    aligned_face = face_align.norm_crop(image, landmark=face.landmark, image_size=crop_size)
    return aligned_face

def get_face_reference(source_image: np.ndarray, reference_face_position: int = 0) -> Optional[np.ndarray]:
    source_faces = find_faces(source_image)
    if not source_faces or reference_face_position >= len(source_faces):
        return None
    
    return extract_face(source_image, source_faces[reference_face_position])

def set_batch_size(size: int):
    """Set the batch size for processing"""
    global BATCH_SIZE
    BATCH_SIZE = max(1, size)
    
def enhance_face(face: np.ndarray) -> np.ndarray:
    """Apply enhancements to face before swapping for better quality"""
    # Convert to float for processing
    face_float = face.astype(np.float32) / 255.0
    
    # Simple contrast enhancement
    enhanced = np.clip(face_float * 1.1, 0, 1) * 255
    
    return enhanced.astype(np.uint8)

def release_resources():
    """Properly release GPU resources"""
    global FACE_ANALYSER, FACE_SWAP_MODEL
    
    FACE_ANALYSER = None
    FACE_SWAP_MODEL = None
    
    # Clean CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

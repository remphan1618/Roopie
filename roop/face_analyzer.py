from typing import Dict, List, Set, Tuple, Union
import warnings
import cv2
import numpy as np
import os
import insightface
from insightface.app.common import Face as InsightFaceFace
import torch

from modules.globals import Global
from modules.face import Face

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
ANALYSIS_MODEL = None
GENDER_AGE_MODEL = None

def get_analysis_model():
    global ANALYSIS_MODEL
    if ANALYSIS_MODEL is None:
        model_path = os.path.join(MODELS_DIR, 'buffalo_l')
        ANALYSIS_MODEL = insightface.app.FaceAnalysis(
            name='buffalo_l',
            root=os.path.dirname(model_path),
            providers=Global.execution_providers
        )
        ANALYSIS_MODEL.prepare(ctx_id=0, det_size=(640, 640))
    return ANALYSIS_MODEL

def get_gender_age_model():
    # Utilize a separate model for gender/age detection if needed
    # This could be loaded on-demand to optimize memory usage
    return get_analysis_model()

def get_one_face(frame: np.ndarray, face_index: int = 0) -> Union[Face, None]:
    """
    Get a specific face from the frame based on index.
    Optimized to use cached results when possible.
    """
    faces = get_many_faces(frame)
    if face_index < len(faces):
        return faces[face_index]
    return None

def get_many_faces(frame: np.ndarray) -> List[Face]:
    """
    Get all faces from the frame.
    Returns faces with the largest areas first.
    """
    try:
        with torch.cuda.amp.autocast(enabled=Global.fp16_enabled):
            model = get_analysis_model()
            # Convert to RGB for model input
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            elif frame.shape[2] == 3 and frame.dtype == np.uint8:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
            detected_faces = model.get(frame, max_num=0)
            
            faces = []
            for detected_face in detected_faces:
                face = Face(
                    detected_face.bbox,
                    detected_face.kps,
                    detected_face.det_score,
                    detected_face.gender,
                    detected_face.age,
                    detected_face.embedding
                )
                faces.append(face)
            # Sort by area (largest first)
            return sorted(faces, key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]), reverse=True)
    except Exception as e:
        warnings.warn(f"Face detection error: {e}")
        return []

def extract_face_images(frame: np.ndarray, faces: List[Face]) -> Dict[int, np.ndarray]:
    """
    Extract face images from the frame using detected faces.
    Returns a dictionary of face index to face image.
    """
    face_images = {}
    for i, face in enumerate(faces):
        # Get face bounding box
        bbox = face.bbox.astype(int)
        # Ensure bbox is within frame
        bbox[0] = max(0, bbox[0])
        bbox[1] = max(0, bbox[1])
        bbox[2] = min(frame.shape[1], bbox[2])
        bbox[3] = min(frame.shape[0], bbox[3])
        # Extract face image
        face_img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        face_images[i] = face_img
    return face_images

def get_face_embeddings(faces: List[Face]) -> List[np.ndarray]:
    """Extract embeddings from detected faces for comparison."""
    return [face.embedding for face in faces]

def compare_faces(faces1: List[Face], faces2: List[Face], threshold: float = 0.7) -> List[Tuple[int, int, float]]:
    """
    Compare faces between two lists and return matching pairs.
    Returns list of tuples (index1, index2, similarity_score).
    """
    matches = []
    # Extract embeddings
    embeddings1 = get_face_embeddings(faces1)
    embeddings2 = get_face_embeddings(faces2)
    
    # Use GPU tensors for faster computation if available
    if Global.cuda_available and len(embeddings1) > 0 and len(embeddings2) > 0:
        embeddings1_tensor = torch.tensor(np.array(embeddings1), device='cuda')
        embeddings2_tensor = torch.tensor(np.array(embeddings2), device='cuda')
        
        # Normalize embeddings
        embeddings1_tensor = torch.nn.functional.normalize(embeddings1_tensor, p=2, dim=1)
        embeddings2_tensor = torch.nn.functional.normalize(embeddings2_tensor, p=2, dim=1)
        
        # Compute cosine similarity
        similarity = torch.mm(embeddings1_tensor, embeddings2_tensor.t()).cpu().numpy()
        
        # Find matches
        for i in range(similarity.shape[0]):
            for j in range(similarity.shape[1]):
                if similarity[i, j] > threshold:
                    matches.append((i, j, similarity[i, j]))
    else:
        # Fallback to CPU computation
        for i, emb1 in enumerate(embeddings1):
            for j, emb2 in enumerate(embeddings2):
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                if similarity > threshold:
                    matches.append((i, j, similarity))
                    
    # Sort by similarity score (highest first)
    return sorted(matches, key=lambda x: x[2], reverse=True)

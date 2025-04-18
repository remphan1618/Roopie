import numpy as np
from typing import Tuple, List, Optional, Union

class Face:
    """
    A class to store face information for efficient processing.
    Includes optimizations for memory usage and faster calculations.
    """
    
    def __init__(self, 
                 bbox: np.ndarray, 
                 kps: np.ndarray, 
                 det_score: float,
                 gender: Union[float, int] = 0, 
                 age: float = 0, 
                 embedding: Optional[np.ndarray] = None,
                 mask: Optional[np.ndarray] = None):
        """
        Initialize face with detected features.
        
        Args:
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            kps: Facial keypoints
            det_score: Detection confidence score
            gender: Gender prediction (0=female, 1=male)
            age: Age prediction
            embedding: Face embedding for recognition
            mask: Optional face mask for blending
        """
        # Ensure bbox and kps are numpy arrays with correct dtype for faster processing
        self.bbox = np.array(bbox, dtype=np.float32)
        self.kps = np.array(kps, dtype=np.float32)
        self.det_score = float(det_score)
        self.gender = gender
        self.age = float(age)
        
        # Store embedding as float16 if provided (saves memory)
        if embedding is not None:
            self.embedding = np.array(embedding, dtype=np.float16)
        else:
            self.embedding = None
            
        # Store mask if provided
        self.mask = mask
        
        # Cache for performance
        self._center = None
        self._size = None
        self._area = None
        
    @property
    def center(self) -> Tuple[float, float]:
        """Calculate center point of face bbox (cached)"""
        if self._center is None:
            self._center = (
                (self.bbox[0] + self.bbox[2]) / 2,
                (self.bbox[1] + self.bbox[3]) / 2
            )
        return self._center
    
    @property
    def size(self) -> Tuple[float, float]:
        """Calculate size of face bbox (cached)"""
        if self._size is None:
            self._size = (
                self.bbox[2] - self.bbox[0],
                self.bbox[3] - self.bbox[1]
            )
        return self._size
        
    @property
    def area(self) -> float:
        """Calculate area of face bbox (cached)"""
        if self._area is None:
            width, height = self.size
            self._area = width * height
        return self._area
    
    def scaled_bbox(self, scale_factor: float) -> np.ndarray:
        """
        Return scaled bounding box by given factor.
        Useful for creating slightly larger face regions.
        """
        center_x, center_y = self.center
        width, height = self.size
        
        new_width = width * scale_factor
        new_height = height * scale_factor
        
        x1 = center_x - new_width / 2
        y1 = center_y - new_height / 2
        x2 = center_x + new_width / 2
        y2 = center_y + new_height / 2
        
        return np.array([x1, y1, x2, y2], dtype=np.float32)
    
    def to_dict(self) -> dict:
        """Convert face to dictionary representation"""
        face_dict = {
            'bbox': self.bbox.tolist(),
            'kps': self.kps.tolist(),
            'det_score': self.det_score,
            'gender': self.gender,
            'age': self.age
        }
        
        if self.embedding is not None:
            face_dict['embedding'] = self.embedding.tolist()
            
        return face_dict
    
    @classmethod
    def from_dict(cls, face_dict: dict) -> 'Face':
        """Create face from dictionary representation"""
        embedding = face_dict.get('embedding')
        if embedding is not None:
            embedding = np.array(embedding, dtype=np.float16)
            
        return cls(
            bbox=np.array(face_dict['bbox'], dtype=np.float32),
            kps=np.array(face_dict['kps'], dtype=np.float32),
            det_score=face_dict['det_score'],
            gender=face_dict.get('gender', 0),
            age=face_dict.get('age', 0),
            embedding=embedding
        )
    
    def is_similar_to(self, other_face: 'Face', threshold: float = 0.7) -> bool:
        """
        Check if this face is similar to another face using embedding similarity.
        
        Args:
            other_face: Another Face object to compare with
            threshold: Similarity threshold (0-1), higher means more similar
            
        Returns:
            Boolean indicating if faces are similar
        """
        if self.embedding is None or other_face.embedding is None:
            return False
            
        # Convert to float32 for calculation to maintain precision
        emb1 = self.embedding.astype(np.float32)
        emb2 = other_face.embedding.astype(np.float32)
        
        # Normalize embeddings
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb2 / np.linalg.norm(emb2)
        
        # Compute cosine similarity
        similarity = np.dot(emb1, emb2)
        
        return similarity > threshold

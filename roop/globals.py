import os
import torch
import platform
import tempfile
from typing import List, Optional, Dict, Any

class Global:
    # Execution providers - CUDA with optional TensorRT
    cuda_available = torch.cuda.is_available()
    execution_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda_available else ['CPUExecutionProvider']
    
    # CUDA device information
    device_count = torch.cuda.device_count() if cuda_available else 0
    current_device = torch.cuda.current_device() if cuda_available else -1
    device_name = torch.cuda.get_device_name(current_device) if cuda_available else "CPU"
    
    # Memory optimizations
    fp16_enabled = cuda_available  # Enable FP16 if CUDA is available
    
    # Platform and system info
    system_platform = platform.system()
    is_windows = system_platform == 'Windows'
    is_linux = system_platform == 'Linux'
    is_macos = system_platform == 'Darwin'
    
    # Temporary directories with cleanup management
    temp_dir = tempfile.mkdtemp()
    temp_frame_dir = os.path.join(temp_dir, 'frames')
    
    # Maximum batch sizes based on VRAM availability
    @classmethod
    def get_optimal_batch_size(cls) -> int:
        """
        Determine the optimal batch size based on available VRAM
        """
        if not cls.cuda_available:
            return 1
            
        try:
            # Get total VRAM in GB
            total_vram = torch.cuda.get_device_properties(cls.current_device).total_memory / (1024**3)
            
            # Set batch size based on VRAM availability
            if total_vram >= 20.0:  # For high-end cards like 3090 (24GB)
                return 16
            elif total_vram >= 10.0:  # For mid-range cards (12-16GB)
                return 8
            elif total_vram >= 6.0:  # For lower-end cards (8GB)
                return 4
            else:  # For minimal VRAM
                return 2
        except:
            return 2  # Default fallback
            
    @classmethod
    def initialize_temp_directories(cls):
        """
        Create temporary directories if they don't exist
        """
        os.makedirs(cls.temp_frame_dir, exist_ok=True)
        
    @classmethod
    def cleanup_temp_directories(cls):
        """
        Clean up temporary directories when done
        """
        if os.path.exists(cls.temp_dir):
            try:
                import shutil
                shutil.rmtree(cls.temp_dir)
            except:
                pass
                
    @classmethod
    def optimize_for_device(cls):
        """
        Apply device-specific optimizations
        """
        if cls.cuda_available:
            # Set tensor cores usage for faster computation if available
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Enable cudnn benchmark for potentially faster runtime
            torch.backends.cudnn.benchmark = True
            
            # Configure for best performance on detected GPU
            if "3090" in cls.device_name or "RTX 30" in cls.device_name:
                # Specific optimizations for RTX 3090
                cls.fp16_enabled = True
                
                # Set higher batch sizes for video processing
                video_processing_batch_size = 8
                
                # Reserve some VRAM for the drivers and system
                try:
                    torch.cuda.set_per_process_memory_fraction(0.9)
                except:
                    pass
            
            # Additional CUDA optimizations
            torch.cuda.empty_cache()

import cv2
import threading
import numpy as np
import onnxruntime
import os
import gdown
import torch
from typing import List, Dict, Optional, Any, Tuple
import gradio as gr
from concurrent.futures import ThreadPoolExecutor

from modules.face_analyzer import get_one_face, get_many_faces
from modules.face_processor import FaceProcessor
from modules.globals import Global
from modules.face import Face
import modules.globals as shared

# Constants for model paths
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
INSWAPPER_MODEL = os.path.join(MODELS_DIR, "inswapper_128.onnx")
INSWAPPER_FP16_MODEL = os.path.join(MODELS_DIR, "inswapper_128.fp16.onnx")

class FaceSwapTab:
    def __init__(self):
        self.face_processor = FaceProcessor()
        self.source_face = None
        self.target_faces = []
        self.result_image = None
        self.processing_thread = None
        self.swap_mode = "swap_first"  # Default to swapping the first detected face
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.model_type = "fp32"  # Default to FP32 model
        
    def download_models(self, progress=None):
        """Download required models if they don't exist"""
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        # Download FP32 model if it doesn't exist
        if not os.path.exists(INSWAPPER_MODEL):
            url = "https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx"
            gdown.download(url, INSWAPPER_MODEL, quiet=False)
            
        # Download FP16 model if it doesn't exist
        if not os.path.exists(INSWAPPER_FP16_MODEL):
            url = "https://huggingface.co/netrunner-exe/Insight-Swap-models/resolve/main/inswapper_128.fp16.onnx"
            gdown.download(url, INSWAPPER_FP16_MODEL, quiet=False)
            
        if progress is not None:
            progress(1.0)
        return "Models downloaded successfully!"
        
    def set_fp16_model(self):
        """Switch to using the FP16 model"""
        self.model_type = "fp16"
        # Update the face processor to use the FP16 model
        if hasattr(self.face_processor, 'inswapper'):
            self.face_processor.inswapper = None  # Force reload with new model
        return "Switched to FP16 model. This will be used for future swaps."
        
    def set_fp32_model(self):
        """Switch to using the FP32 model"""
        self.model_type = "fp32"
        # Update the face processor to use the FP32 model
        if hasattr(self.face_processor, 'inswapper'):
            self.face_processor.inswapper = None  # Force reload with new model
        return "Switched to FP32 model. This will be used for future swaps."
    
    def load_model(self):
        """Load the appropriate model based on current settings"""
        # This will be called by the face processor when needed
        model_path = INSWAPPER_FP16_MODEL if self.model_type == "fp16" else INSWAPPER_MODEL
        
        # Configure providers based on CUDA availability and model type
        providers = Global.execution_providers
        
        # Set execution mode for optimal performance
        sess_options = onnxruntime.SessionOptions()
        
        # Configure for CUDA if available
        if Global.cuda_available:
            # Enable memory optimizations
            sess_options.enable_cpu_mem_arena = False
            sess_options.enable_mem_pattern = False
            sess_options.enable_mem_reuse = True
            
            # Set execution mode
            sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
            sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Configure CUDA provider options
            provider_options = [
                {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': int(0.8 * (torch.cuda.get_device_properties(0).total_memory)),
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }
            ]
            
            # Create session with optimized configuration
            model = onnxruntime.InferenceSession(
                model_path, 
                sess_options=sess_options,
                providers=[('CUDAExecutionProvider', provider_options[0]), 'CPUExecutionProvider']
            )
        else:
            # CPU configuration
            sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
            model = onnxruntime.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=['CPUExecutionProvider']
            )
            
        return model
        
    def set_source_image(self, image, face_index=0):
        """Process the source image and extract the face"""
        if image is None:
            return None, "No source image provided."
            
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            
        # Extract face with CUDA acceleration if available
        with torch.cuda.amp.autocast(enabled=Global.fp16_enabled):
            self.source_face = get_one_face(image, face_index)
            
        if self.source_face is None:
            return None, "No face detected in source image."
            
        # Draw rectangle around the detected face
        result_image = image.copy()
        bbox = self.source_face.bbox.astype(int)
        cv2.rectangle(result_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        
        return result_image, "Source face detected successfully!"
        
    def set_target_image(self, image):
        """Process the target image and detect all faces"""
        if image is None:
            return None, "No target image provided."
            
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            
        # Detect all faces in the target image
        with torch.cuda.amp.autocast(enabled=Global.fp16_enabled):
            self.target_faces = get_many_faces(image)
            
        if not self.target_faces:
            return None, "No faces detected in target image."
            
        # Draw rectangles around all detected faces
        result_image = image.copy()
        for i, face in enumerate(self.target_faces):
            bbox = face.bbox.astype(int)
            # Different color for different face indices
            color = (0, 255, 0) if i == 0 else (255, 0, 0)
            cv2.rectangle(result_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            # Put face index
            cv2.putText(result_image, f"Face {i}", (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        
        return result_image, f"Detected {len(self.target_faces)} faces in target image."
        
    def swap_face(self, image=None, source_face=None, target_faces=None, swap_mode="swap_first"):
        """
        Perform face swapping based on swap mode
        
        Args:
            image: Target image (optional, uses stored target if None)
            source_face: Source face (optional, uses stored source if None)
            target_faces: Target faces (optional, uses stored targets if None)
            swap_mode: Swap mode - "swap_first", "swap_all", or "face_index_{i}"
        """
        if image is None or (source_face is None and self.source_face is None) or (target_faces is None and not self.target_faces):
            return None, "Missing required inputs for face swap."
            
        # Use provided inputs or stored inputs
        source_face = source_face if source_face is not None else self.source_face
        target_faces = target_faces if target_faces is not None else self.target_faces
        
        # Parse swap mode
        if swap_mode.startswith("face_index_"):
            try:
                face_index = int(swap_mode.split("_")[-1])
                if face_index < 0 or face_index >= len(target_faces):
                    return None, f"Invalid face index {face_index}. Only {len(target_faces)} faces detected."
                faces_to_swap = [target_faces[face_index]]
            except:
                faces_to_swap = [target_faces[0]]
        elif swap_mode == "swap_all":
            faces_to_swap = target_faces
        else:  # Default to swap_first
            faces_to_swap = [target_faces[0]] if target_faces else []
            
        if not faces_to_swap:
            return None, "No faces to swap."
            
        # When using the FP16 model, customize the face processor
        if self.model_type == "fp16":
            # Inject the FP16 model into the processor
            if self.face_processor.inswapper is None:
                self.face_processor.inswapper = self.load_model()
                
            # Set global FP16 flag to True
            Global.fp16_enabled = True
            
        # Perform face swapping with optimized settings
        try:
            with torch.cuda.amp.autocast(enabled=Global.fp16_enabled):
                result_image = image.copy()
                
                # Apply batch processing if multiple faces
                if len(faces_to_swap) > 1:
                    batch_size = min(len(faces_to_swap), Global.get_optimal_batch_size())
                    for i in range(0, len(faces_to_swap), batch_size):
                        batch_faces = faces_to_swap[i:i+batch_size]
                        # Process each face in the batch
                        for face in batch_faces:
                            result_image = self.face_processor.process_frame(
                                source_face,
                                result_image,
                                [face],
                                enhance_face=False,
                                swap_all=False
                            )
                else:
                    # Single face processing
                    result_image = self.face_processor.process_frame(
                        source_face,
                        result_image,
                        faces_to_swap,
                        enhance_face=False,
                        swap_all=True
                    )
                    
                self.result_image = result_image
                return result_image, "Face swap completed successfully!"
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"Error during face swap: {str(e)}"
            
    def create_ui_tabs(self):
        """Create Gradio UI for the face swap tab"""
        with gr.Tab("Face Swap"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Source Face")
                    source_image = gr.Image(label="Source Image")
                    source_face_index = gr.Number(value=0, label="Source Face Index", min_value=0, step=1)
                    process_source_btn = gr.Button("Process Source")
                    source_result = gr.Image(label="Source Result")
                    source_status = gr.Textbox(label="Source Status")
                    
                with gr.Column():
                    gr.Markdown("### Target Image")
                    target_image = gr.Image(label="Target Image")
                    process_target_btn = gr.Button("Process Target")
                    target_result = gr.Image(label="Target Result")
                    target_status = gr.Textbox(label="Target Status")
                    
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Model Settings")
                    with gr.Row():
                        download_models_btn = gr.Button("Download Models")
                        model_download_status = gr.Textbox(label="Download Status")
                    with gr.Row():
                        use_fp16_btn = gr.Button("Use FP16 Model (Faster)")
                        use_fp32_btn = gr.Button("Use FP32 Model (More Compatible)")
                    model_status = gr.Textbox(label="Model Status")
                    
                with gr.Column():
                    gr.Markdown("### Swap Settings")
                    swap_mode = gr.Radio(
                        ["swap_first", "swap_all"] + [f"face_index_{i}" for i in range(10)],
                        value="swap_first",
                        label="Swap Mode"
                    )
                    swap_btn = gr.Button("Swap Face")
                    
            with gr.Row():
                gr.Markdown("### Result")
                result_image = gr.Image(label="Result Image")
                result_status = gr.Textbox(label="Swap Status")
                
            # Connect UI components to functions
            process_source_btn.click(
                fn=self.set_source_image,
                inputs=[source_image, source_face_index],
                outputs=[source_result, source_status]
            )
            
            process_target_btn.click(
                fn=self.set_target_image,
                inputs=[target_image],
                outputs=[target_result, target_status]
            )
            
            download_models_btn.click(
                fn=self.download_models,
                inputs=[],
                outputs=[model_download_status]
            )
            
            use_fp16_btn.click(
                fn=self.set_fp16_model,
                inputs=[],
                outputs=[model_status]
            )
            
            use_fp32_btn.click(
                fn=self.set_fp32_model,
                inputs=[],
                outputs=[model_status]
            )
            
            swap_btn.click(
                fn=self.swap_face,
                inputs=[target_image, swap_mode],
                outputs=[result_image, result_status]
            )
            
    def initialize(self):
        """Initialize the tab and ensure models are available"""
        # Check for CUDA availability and optimize settings
        if Global.cuda_available:
            Global.optimize_for_device()
            # If we have an RTX 3090, default to FP16 model
            if "3090" in Global.device_name:
                self.model_type = "fp16"
                Global.fp16_enabled = True
                
        # Ensure models are downloaded
        if not os.path.exists(INSWAPPER_MODEL) or not os.path.exists(INSWAPPER_FP16_MODEL):
            self.download_models()
            
        return "Face Swap tab initialized successfully!"

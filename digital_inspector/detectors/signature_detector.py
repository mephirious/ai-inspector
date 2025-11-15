"""Signature detection using YOLOv8 model."""

import numpy as np
from typing import List, Dict, Any, Optional
from ultralytics import YOLO
import torch
import os

try:
    from ..utils.model_utils import get_model_path
except ImportError:
    from digital_inspector.utils.model_utils import get_model_path


class SignatureDetector:
    """Detector for signatures using YOLOv8 model."""
    
    def __init__(
        self,
        model_path: str = "obazl/yolov8-signature-detection",
        device: str = "auto",
        confidence_threshold: float = 0.25,
        token: Optional[str] = None
    ):
        """
        Initialize signature detector.
        
        Args:
            model_path: Path to YOLOv8 model (HuggingFace repo or local path)
            device: Device to use ('cpu', 'cuda', or 'auto')
            confidence_threshold: Minimum confidence threshold
            token: HuggingFace token for gated repositories (or set HF_TOKEN env var)
        """
        self.confidence_threshold = confidence_threshold
        
        # Auto-detect device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load model
        try:
            # Check if model_path is a HuggingFace repo or local path
            if "/" in model_path and not os.path.exists(model_path) and not model_path.endswith(".pt"):
                # Likely a HuggingFace repo, download it
                print(f"Downloading model from HuggingFace: {model_path}")
                model_path = get_model_path(model_path, token=token)
                print(f"Model downloaded to: {model_path}")
            
            self.model = YOLO(model_path)
            self.model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load signature model from {model_path}: {str(e)}")
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect signatures in image.
        
        Args:
            image: Input image as numpy array (H, W, C) in RGB format
        
        Returns:
            List of detection dictionaries with keys:
            - category: "signature"
            - bbox: [x1, y1, x2, y2] in absolute pixels
            - confidence: float
        """
        try:
            # Run inference
            results = self.model(
                image,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False
            )
            
            detections = []
            if results and len(results) > 0:
                result = results[0]
                
                # Extract boxes, scores, classes
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                    scores = result.boxes.conf.cpu().numpy()
                    
                    for box, score in zip(boxes, scores):
                        detections.append({
                            "category": "signature",
                            "bbox": box.tolist(),  # [x1, y1, x2, y2]
                            "confidence": float(score)
                        })
            
            return detections
        
        except Exception as e:
            print(f"Error in signature detection: {str(e)}")
            return []


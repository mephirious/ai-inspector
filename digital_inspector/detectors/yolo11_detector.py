"""YOLO11 detector for Digital Inspector (fine-tuned model)."""

import numpy as np
from typing import List, Dict, Any, Optional
from ultralytics import YOLO
import torch
import os
from pathlib import Path


# Class mapping (must match training)
CLASS_NAMES = ["signature", "stamp", "qr"]


class YOLO11Detector:
    """Unified YOLO11 detector for signatures, stamps, and QR codes."""
    
    def __init__(
        self,
        model_path: str = None,
        device: str = "auto",
        confidence_threshold: float = 0.25
    ):
        """
        Initialize YOLO11 detector.
        
        Args:
            model_path: Path to fine-tuned model (best.pt or best.onnx)
                       If None, looks for model in training/runs directory
            device: Device to use ('cpu', 'cuda', or 'auto')
            confidence_threshold: Minimum confidence threshold
        """
        self.confidence_threshold = confidence_threshold
        
        # Auto-detect device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Find model if not provided
        if model_path is None:
            model_path = self._find_model()
        
        if model_path is None:
            raise FileNotFoundError(
                "No model found. Please train a model first or specify model_path."
            )
        
        # Load model
        try:
            print(f"Loading YOLO11 model from: {model_path}")
            self.model = YOLO(model_path)
            self.model.to(self.device)
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO11 model from {model_path}: {str(e)}")
    
    def _find_model(self) -> Optional[str]:
        """Find the best trained model in runs directory."""
        project_root = Path(__file__).parent.parent.parent
        runs_dir = project_root / "training" / "runs" / "yolo11_finetuned"
        
        # Try to find best.pt
        best_pt = runs_dir / "weights" / "best.pt"
        if best_pt.exists():
            return str(best_pt)
        
        # Try to find best.onnx
        best_onnx = runs_dir / "weights" / "best.onnx"
        if best_onnx.exists():
            return str(best_onnx)
        
        # Try to find last.pt
        last_pt = runs_dir / "weights" / "last.pt"
        if last_pt.exists():
            print("Warning: Using last.pt instead of best.pt")
            return str(last_pt)
        
        return None
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect signatures, stamps, and QR codes in image.
        
        Args:
            image: Input image as numpy array (H, W, C) in RGB format
        
        Returns:
            List of detection dictionaries with keys:
            - category: "signature", "stamp", or "qr"
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
                    classes = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for box, score, cls in zip(boxes, scores, classes):
                        # Map class ID to category name
                        if 0 <= cls < len(CLASS_NAMES):
                            category = CLASS_NAMES[cls]
                        else:
                            category = "unknown"
                        
                        detections.append({
                            "category": category,
                            "bbox": box.tolist(),  # [x1, y1, x2, y2]
                            "confidence": float(score)
                        })
            
            return detections
        
        except Exception as e:
            print(f"Error in YOLO11 detection: {str(e)}")
            return []


def create_unified_detector(
    model_path: str = None,
    device: str = "auto",
    confidence_threshold: float = 0.25
) -> YOLO11Detector:
    """
    Create a unified YOLO11 detector that detects all three classes.
    
    This replaces the need for separate SignatureDetector, StampDetector, etc.
    """
    return YOLO11Detector(
        model_path=model_path,
        device=device,
        confidence_threshold=confidence_threshold
    )


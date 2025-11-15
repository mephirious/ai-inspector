"""QR code detection using qrdet and pyzbar."""

import numpy as np
from typing import List, Dict, Any, Optional
try:
    from qrdet import QRDetector as QRDetectorModel
except ImportError:
    QRDetectorModel = None

try:
    from pyzbar import decode as pyzbar_decode
    import cv2
    PYZBAR_AVAILABLE = True
except ImportError:
    pyzbar_decode = None
    cv2 = None
    PYZBAR_AVAILABLE = False


class QRDetector:
    """Detector for QR codes using qrdet and pyzbar."""
    
    def __init__(
        self,
        confidence_threshold: float = 0.3,
        use_decoder: bool = True
    ):
        """
        Initialize QR detector.
        
        Args:
            confidence_threshold: Minimum confidence threshold
            use_decoder: Whether to decode QR codes using pyzbar
        """
        self.confidence_threshold = confidence_threshold
        self.use_decoder = use_decoder and PYZBAR_AVAILABLE
        
        # Initialize qrdet model
        if QRDetectorModel is None:
            raise ImportError(
                "qrdet package not found. Install with: pip install qrdet"
            )
        
        try:
            self.model = QRDetectorModel()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize QR detector: {str(e)}")
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect QR codes in image.
        
        Args:
            image: Input image as numpy array (H, W, C) in RGB format
        
        Returns:
            List of detection dictionaries with keys:
            - category: "qr"
            - bbox: [x1, y1, x2, y2] in absolute pixels
            - confidence: float
            - data: decoded QR data (if available)
        """
        try:
            detections = []
            
            # Convert RGB to BGR for qrdet (if needed)
            if len(image.shape) == 3:
                # qrdet typically expects BGR or RGB, try RGB first
                image_input = image.copy()
            else:
                image_input = image
            
            # Run qrdet detection
            try:
                # qrdet API: detect() returns list of QR codes
                results = self.model.detect(image_input)
                
                if results and len(results) > 0:
                    for result in results:
                        # qrdet returns dict with 'bbox_xyxy' or 'bbox' key
                        if isinstance(result, dict):
                            bbox = result.get('bbox_xyxy', result.get('bbox', result.get('box', [])))
                            confidence = result.get('confidence', result.get('score', 0.9))
                        elif hasattr(result, 'bbox_xyxy'):
                            bbox = result.bbox_xyxy
                            confidence = getattr(result, 'confidence', 0.9)
                        elif hasattr(result, 'bbox'):
                            bbox = result.bbox
                            confidence = getattr(result, 'confidence', 0.9)
                        elif isinstance(result, (list, tuple)) and len(result) >= 4:
                            bbox = list(result[:4])
                            confidence = 0.9
                        else:
                            continue
                        
                        # Ensure bbox is [x1, y1, x2, y2]
                        if len(bbox) == 4:
                            if confidence >= self.confidence_threshold:
                                detection = {
                                    "category": "qr",
                                    "bbox": bbox,
                                    "confidence": float(confidence)
                                }
                                
                                # Try to decode QR code
                                if self.use_decoder:
                                    decoded_data = self._decode_qr(image, bbox)
                                    if decoded_data:
                                        detection["data"] = decoded_data
                                
                                detections.append(detection)
            
            except Exception as e:
                # Fallback: try pyzbar directly
                if self.use_decoder:
                    detections = self._detect_with_pyzbar(image)
            
            return detections
        
        except Exception as e:
            print(f"Error in QR detection: {str(e)}")
            return []
    
    def _decode_qr(self, image: np.ndarray, bbox: List[float]) -> Optional[str]:
        """Decode QR code from image region using pyzbar."""
        if not self.use_decoder or not PYZBAR_AVAILABLE:
            return None
        
        try:
            # Extract ROI
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            
            roi = image[y1:y2, x1:x2]
            
            # Convert to grayscale for pyzbar
            if len(roi.shape) == 3:
                if roi.shape[2] == 3:
                    # RGB to grayscale
                    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
                else:
                    gray = roi[:, :, 0]
            else:
                gray = roi
            
            # Decode
            decoded = pyzbar_decode(gray)
            if decoded and len(decoded) > 0:
                return decoded[0].data.decode('utf-8', errors='ignore')
        
        except Exception:
            pass
        
        return None
    
    def _detect_with_pyzbar(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Fallback detection using pyzbar only."""
        if not PYZBAR_AVAILABLE or cv2 is None:
            return []
        
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                if image.shape[2] == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                else:
                    gray = image[:, :, 0]
            else:
                gray = image
            
            # Decode
            decoded = pyzbar_decode(gray)
            
            detections = []
            for obj in decoded:
                # Get bounding box
                rect = obj.rect
                bbox = [rect.left, rect.top, rect.left + rect.width, rect.top + rect.height]
                
                detections.append({
                    "category": "qr",
                    "bbox": bbox,
                    "confidence": 0.9,  # pyzbar doesn't provide confidence
                    "data": obj.data.decode('utf-8', errors='ignore')
                })
            
            return detections
        
        except Exception:
            return []


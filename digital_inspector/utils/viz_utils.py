"""Visualization utilities for drawing bounding boxes on images."""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple


# Color mapping for different categories
COLOR_MAP = {
    'signature': (255, 0, 0),      # Blue (BGR)
    'stamp': (0, 0, 255),          # Red (BGR)
    'qr': (0, 255, 0),             # Green (BGR)
    'barcode': (0, 165, 255),      # Orange (BGR)
    'default': (255, 255, 255)     # White (BGR)
}


def draw_bbox(
    image: np.ndarray,
    bbox: List[float],
    label: str,
    confidence: float,
    color: Tuple[int, int, int] = None
) -> np.ndarray:
    """
    Draw a single bounding box on image.
    
    Args:
        image: Input image as numpy array (BGR format)
        bbox: Bounding box [x, y, width, height] in pixels
        label: Label text
        confidence: Confidence score
        color: BGR color tuple, or None to use default for category
    
    Returns:
        Image with bounding box drawn
    """
    x, y, w, h = bbox
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)
    
    if color is None:
        color = COLOR_MAP.get(label.lower(), COLOR_MAP['default'])
    
    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    
    # Draw label background
    label_text = f"{label}: {confidence:.2f}"
    (text_width, text_height), baseline = cv2.getTextSize(
        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )
    
    cv2.rectangle(
        image,
        (x1, y1 - text_height - baseline - 5),
        (x1 + text_width, y1),
        color,
        -1
    )
    
    # Draw label text
    cv2.putText(
        image,
        label_text,
        (x1, y1 - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )
    
    return image


def draw_detections(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    page_width: int,
    page_height: int
) -> np.ndarray:
    """
    Draw all detections on image.
    
    Args:
        image: Input image as numpy array (BGR format)
        detections: List of detection dictionaries
        page_width: Original page width (for denormalizing bboxes)
        page_height: Original page height (for denormalizing bboxes)
    
    Returns:
        Annotated image
    """
    annotated = image.copy()
    
    for det in detections:
        bbox_norm = det.get('bbox', [])
        if len(bbox_norm) != 4:
            continue
        
        # Denormalize bbox
        x_norm, y_norm, w_norm, h_norm = bbox_norm
        bbox = [
            x_norm * page_width,
            y_norm * page_height,
            w_norm * page_width,
            h_norm * page_height
        ]
        
        category = det.get('category', 'unknown')
        confidence = det.get('confidence', 0.0)
        
        annotated = draw_bbox(annotated, bbox, category, confidence)
    
    return annotated


def save_annotated_image(
    image: np.ndarray,
    output_path: str,
    detections: List[Dict[str, Any]],
    page_width: int,
    page_height: int
):
    """
    Save image with annotations.
    
    Args:
        image: Input image as numpy array (RGB or BGR)
        output_path: Path to save annotated image
        detections: List of detection dictionaries
        page_width: Original page width
        page_height: Original page height
    """
    # Ensure BGR format for OpenCV
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Check if RGB (common PIL format) and convert to BGR
        if image.dtype == np.uint8:
            # Assume RGB, convert to BGR
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
    else:
        image_bgr = image
    
    annotated = draw_detections(image_bgr, detections, page_width, page_height)
    cv2.imwrite(output_path, annotated)


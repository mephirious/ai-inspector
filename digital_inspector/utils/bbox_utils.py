"""Bounding box conversion and manipulation utilities."""

from typing import List, Tuple, Dict
import numpy as np


def yolo_to_xywh(bbox: List[float], image_width: int, image_height: int) -> List[float]:
    """
    Convert YOLO format [x_center, y_center, width, height] (normalized)
    to [x, y, width, height] (absolute pixels).
    
    Args:
        bbox: YOLO bbox [x_center, y_center, width, height] (0-1 normalized)
        image_width: Image width in pixels
        image_height: Image height in pixels
    
    Returns:
        [x, y, width, height] in absolute pixels
    """
    x_center, y_center, w, h = bbox
    x = (x_center - w / 2) * image_width
    y = (y_center - h / 2) * image_height
    width = w * image_width
    height = h * image_height
    return [x, y, width, height]


def yolo_to_xyxy(bbox: List[float], image_width: int, image_height: int) -> List[float]:
    """
    Convert YOLO format to [x1, y1, x2, y2] (absolute pixels).
    
    Args:
        bbox: YOLO bbox [x_center, y_center, width, height] (0-1 normalized)
        image_width: Image width in pixels
        image_height: Image height in pixels
    
    Returns:
        [x1, y1, x2, y2] in absolute pixels
    """
    x_center, y_center, w, h = bbox
    x1 = (x_center - w / 2) * image_width
    y1 = (y_center - h / 2) * image_height
    x2 = (x_center + w / 2) * image_width
    y2 = (y_center + h / 2) * image_height
    return [x1, y1, x2, y2]


def xyxy_to_xywh(bbox: List[float]) -> List[float]:
    """
    Convert [x1, y1, x2, y2] to [x, y, width, height].
    
    Args:
        bbox: [x1, y1, x2, y2]
    
    Returns:
        [x, y, width, height]
    """
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2 - x1, y2 - y1]


def normalize_bbox(bbox: List[float], page_width: int, page_height: int) -> List[float]:
    """
    Normalize bbox coordinates to [0, 1] range relative to page size.
    
    Args:
        bbox: [x, y, width, height] in absolute pixels
        page_width: Page width in pixels
        page_height: Page height in pixels
    
    Returns:
        [x_norm, y_norm, width_norm, height_norm] normalized to [0, 1]
    """
    x, y, w, h = bbox
    return [
        x / page_width,
        y / page_height,
        w / page_width,
        h / page_height
    ]


def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) for two bboxes.
    
    Args:
        bbox1: [x1, y1, x2, y2]
        bbox2: [x1, y1, x2, y2]
    
    Returns:
        IoU value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


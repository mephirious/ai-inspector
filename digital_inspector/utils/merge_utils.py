"""Merge and deduplicate detection results from multiple detectors."""

from typing import List, Dict, Any
from .bbox_utils import xyxy_to_xywh, normalize_bbox, calculate_iou


def merge_detections(
    all_detections: List[Dict[str, Any]],
    page_width: int,
    page_height: int,
    iou_threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Merge detections from all detectors, remove duplicates, and normalize coordinates.
    
    Args:
        all_detections: List of detection dicts from all detectors
        page_width: Page width in pixels
        page_height: Page height in pixels
        iou_threshold: IoU threshold for duplicate removal
    
    Returns:
        Merged and deduplicated list of detections
    """
    if not all_detections:
        return []
    
    # Convert all bboxes to xyxy format for comparison
    processed = []
    for det in all_detections:
        bbox = det.get('bbox', [])
        if len(bbox) == 4:
            # Assume format is either [x1, y1, x2, y2] or [x, y, w, h]
            # Try to detect format by checking if w/h > image dimensions
            if bbox[2] < page_width and bbox[3] < page_height:
                # Likely [x, y, w, h], convert to xyxy
                x, y, w, h = bbox
                bbox_xyxy = [x, y, x + w, y + h]
            else:
                # Likely already xyxy
                bbox_xyxy = bbox
            
            processed.append({
                'category': det.get('category', 'unknown'),
                'bbox_xyxy': bbox_xyxy,
                'bbox_original': bbox,
                'confidence': det.get('confidence', 0.0),
                'data': det.get('data', None)  # For QR code decoded data
            })
    
    # Remove duplicates using IoU
    merged = []
    used = set()
    
    for i, det1 in enumerate(processed):
        if i in used:
            continue
        
        # Find overlapping detections
        overlaps = [i]
        for j, det2 in enumerate(processed[i+1:], start=i+1):
            if j in used:
                continue
            
            iou = calculate_iou(det1['bbox_xyxy'], det2['bbox_xyxy'])
            if iou > iou_threshold:
                overlaps.append(j)
        
        # Keep detection with highest confidence
        if len(overlaps) > 1:
            best = max(overlaps, key=lambda idx: processed[idx]['confidence'])
            det = processed[best]
        else:
            det = det1
        
        # Convert to final format: [x, y, width, height] normalized
        bbox_xywh = xyxy_to_xywh(det['bbox_xyxy'])
        bbox_normalized = normalize_bbox(bbox_xywh, page_width, page_height)
        
        merged.append({
            'category': det['category'],
            'bbox': bbox_normalized,
            'confidence': det['confidence'],
            'data': det.get('data')
        })
        
        used.update(overlaps)
    
    return merged


def format_output(
    document_name: str,
    page_number: int,
    page_width: int,
    page_height: int,
    detections: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Format final output according to required schema.
    
    Args:
        document_name: Name of the document
        page_number: Page number (1-indexed)
        page_width: Page width in pixels
        page_height: Page height in pixels
        detections: List of merged detections
    
    Returns:
        Formatted output dictionary
    """
    return {
        "document_name": document_name,
        "page_number": page_number,
        "page_size": {
            "width": page_width,
            "height": page_height
        },
        "detections": detections
    }


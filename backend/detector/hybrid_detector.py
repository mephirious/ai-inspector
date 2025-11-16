"""
Hybrid Detector using 3 YOLO models for signature, stamp, and QR code detection.
Maintains the exact detection logic from the original script.
"""
from pathlib import Path
from typing import List, Dict
import cv2
import torch
from ultralytics import YOLO


# Model paths (adjust these if needed)
QR_MODEL_PATH = "/home/mephirious/Projects/ai/qr.pt"
STAMP_MODEL_PATH = "/home/mephirious/Projects/ai/stamp.pt"
SIGNATURE_MODEL_PATH = "/home/mephirious/Projects/ai/signature_best.pt"

# Detection settings
STAMP_CLASS_ID = 15
QR_IMG_SIZES = [640]
STAMP_IMG_SIZES = [640]
SIGNATURE_IMG_SIZES = [640, 1024]

QR_CONF = 0.5
STAMP_CONF = 0.1
SIGNATURE_CONF = 0.1

MERGE_IOU = 0.5


def iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate IoU for bbox in normalized format [x, y, w, h].
    
    Args:
        box1: Normalized bbox [x, y, w, h]
        box2: Normalized bbox [x, y, w, h]
    
    Returns:
        IoU value between 0 and 1
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa1, ya1, xa2, ya2 = x1, y1, x1 + w1, y1 + h1
    xb1, yb1, xb2, yb2 = x2, y2, x2 + w2, y2 + h2

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = w1 * h1
    area_b = w2 * h2
    union = area_a + area_b - inter_area + 1e-9
    return inter_area / union


def merge_detections(dets: List[Dict], iou_thresh: float = MERGE_IOU) -> List[Dict]:
    """
    Merge close detections (keep highest confidence within same category).
    
    Args:
        dets: List of detection dictionaries
        iou_thresh: IoU threshold for merging
    
    Returns:
        Merged list of detections
    """
    if not dets:
        return []

    dets = sorted(dets, key=lambda d: d["confidence"], reverse=True)
    kept: List[Dict] = []

    for d in dets:
        keep = True
        for k in kept:
            if d["category"] != k["category"]:
                continue
            if iou(d["bbox"], k["bbox"]) >= iou_thresh:
                keep = False
                break
        if keep:
            kept.append(d)
    return kept


class HybridDetector3Models:
    """
    Hybrid detector using 3 YOLO models:
      - qr.pt              → category 'qr'
      - stamp.pt           → class STAMP_CLASS_ID → 'stamp'
      - signature_best.pt  → class 0 → 'signature'
    """

    def __init__(self):
        """Initialize all three YOLO models."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[HybridDetector3Models] Loading models on device: {device}")
        self.device = device

        self.qr_model = YOLO(QR_MODEL_PATH)
        self.stamp_model = YOLO(STAMP_MODEL_PATH)
        self.signature_model = YOLO(SIGNATURE_MODEL_PATH)

    def _run_qr(self, image: cv2.typing.MatLike, imgsz: int) -> List[Dict]:
        """
        Run QR code detection model.
        
        Args:
            image: OpenCV image (numpy array)
            imgsz: Image size for inference
        
        Returns:
            List of detection dictionaries
        """
        results = self.qr_model.predict(
            source=image,
            imgsz=imgsz,
            conf=QR_CONF,
            device=self.device,
            verbose=False,
        )
        res = results[0]
        h, w = res.orig_shape

        out: List[Dict] = []
        for box in res.boxes:
            score = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            nx = x1 / w
            ny = y1 / h
            nw = (x2 - x1) / w
            nh = (y2 - y1) / h

            out.append(
                {
                    "category": "qr",
                    "bbox": [nx, ny, nw, nh],
                    "confidence": score,
                    "source": f"qr_{imgsz}",
                }
            )
        print(f"[qr][imgsz={imgsz}] found {len(out)} detections")
        return out

    def _run_stamp(self, image: cv2.typing.MatLike, imgsz: int) -> List[Dict]:
        """
        Run stamp detection model.
        
        Args:
            image: OpenCV image (numpy array)
            imgsz: Image size for inference
        
        Returns:
            List of detection dictionaries
        """
        results = self.stamp_model.predict(
            source=image,
            imgsz=imgsz,
            conf=STAMP_CONF,
            device=self.device,
            classes=[STAMP_CLASS_ID],  # only stamps
            verbose=False,
        )
        res = results[0]
        h, w = res.orig_shape

        out: List[Dict] = []
        for box in res.boxes:
            score = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            nx = x1 / w
            ny = y1 / h
            nw = (x2 - x1) / w
            nh = (y2 - y1) / h

            out.append(
                {
                    "category": "stamp",
                    "bbox": [nx, ny, nw, nh],
                    "confidence": score,
                    "source": f"stamp_{imgsz}",
                }
            )
        print(f"[stamp][imgsz={imgsz}] found {len(out)} detections")
        return out

    def _run_signature(self, image: cv2.typing.MatLike, imgsz: int) -> List[Dict]:
        """
        Run signature detection model.
        signature_best.pt — single class 0 = signature.
        
        Args:
            image: OpenCV image (numpy array)
            imgsz: Image size for inference
        
        Returns:
            List of detection dictionaries
        """
        results = self.signature_model.predict(
            source=image,
            imgsz=imgsz,
            conf=SIGNATURE_CONF,
            device=self.device,
            verbose=False,
        )
        res = results[0]
        h, w = res.orig_shape

        out: List[Dict] = []
        for box in res.boxes:
            score = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            nx = x1 / w
            ny = y1 / h
            nw = (x2 - x1) / w
            nh = (y2 - y1) / h

            out.append(
                {
                    "category": "signature",
                    "bbox": [nx, ny, nw, nh],
                    "confidence": score,
                    "source": f"signature_{imgsz}",
                }
            )
        print(f"[signature][imgsz={imgsz}] found {len(out)} detections")
        return out

    def detect(self, image: cv2.typing.MatLike) -> List[Dict]:
        """
        Run all three models on an image and merge results.
        
        Args:
            image: OpenCV image (numpy array)
        
        Returns:
            List of merged detection dictionaries
        """
        all_dets: List[Dict] = []

        # QR
        for size in QR_IMG_SIZES:
            all_dets.extend(self._run_qr(image, size))

        # Stamp
        for size in STAMP_IMG_SIZES:
            all_dets.extend(self._run_stamp(image, size))

        # Signature
        for size in SIGNATURE_IMG_SIZES:
            all_dets.extend(self._run_signature(image, size))

        merged = merge_detections(all_dets, iou_thresh=MERGE_IOU)
        print(f"[merge] kept {len(merged)} detections after IoU merge")
        return merged

    def annotate_image(
        self, image: cv2.typing.MatLike, detections: List[Dict]
    ) -> cv2.typing.MatLike:
        """
        Draw bounding boxes and labels on image.
        
        Args:
            image: OpenCV image (numpy array)
            detections: List of detection dictionaries
        
        Returns:
            Annotated image
        """
        img = image.copy()
        h, w = img.shape[:2]

        colors = {
            "signature": (255, 0, 0),  # BGR: blue
            "stamp": (0, 0, 255),      # red
            "qr": (0, 255, 0),         # green
        }

        for det in detections:
            x, y, bw, bh = det["bbox"]
            conf = det["confidence"]
            cat = det["category"]

            x1 = int(x * w)
            y1 = int(y * h)
            x2 = int((x + bw) * w)
            y2 = int((y + bh) * h)

            color = colors.get(cat, (255, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{cat} {conf:.2f}"
            cv2.putText(
                img,
                label,
                (x1, max(y1 - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        return img


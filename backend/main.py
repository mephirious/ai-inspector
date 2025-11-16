"""
FastAPI backend for hybrid document detection service.
"""
import os
import uuid
import zipfile
from pathlib import Path
from typing import Dict, List
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from detector.hybrid_detector import HybridDetector3Models
from utils.pdf import pdf_to_images
from utils.images import read_image, is_image_file, is_pdf_file, save_image
from schemas import DetectionResponse, ErrorResponse, PageAnnotations, Annotation, BBox, PageSize


# Initialize FastAPI app with custom JSON encoder for Unicode support
import json as json_lib
from fastapi.encoders import jsonable_encoder

# Custom JSON encoder that doesn't escape Unicode
class UnicodeJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return json_lib.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            separators=(",", ": "),
        ).encode("utf-8")

app = FastAPI(
    title="Hybrid Document Detector API",
    description="3-model hybrid YOLO detector for signatures, stamps, and QR codes",
    version="1.0.0",
    default_response_class=UnicodeJSONResponse
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector (singleton)
detector = None
STORAGE_DIR = Path("/tmp/detector_storage")
STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def get_detector() -> HybridDetector3Models:
    """Get or initialize the detector singleton."""
    global detector
    if detector is None:
        detector = HybridDetector3Models()
    return detector


def normalize_bbox_to_pixels(
    bbox_norm: List[float], img_width: int, img_height: int
) -> Dict[str, float]:
    """
    Convert normalized bbox [x, y, w, h] to pixel coordinates.
    
    Args:
        bbox_norm: Normalized bbox [x, y, w, h]
        img_width: Image width in pixels
        img_height: Image height in pixels
    
    Returns:
        Dictionary with x, y, width, height in pixels
    """
    x_norm, y_norm, w_norm, h_norm = bbox_norm
    x = x_norm * img_width
    y = y_norm * img_height
    width = w_norm * img_width
    height = h_norm * img_height
    return {"x": x, "y": y, "width": width, "height": height}


def process_detections(
    detections: List[Dict], img_width: int, img_height: int
) -> Dict[str, Annotation]:
    """
    Convert detections to annotation format with IDs.
    
    Args:
        detections: List of detection dictionaries
        img_width: Image width in pixels
        img_height: Image height in pixels
    
    Returns:
        Dictionary of annotations with IDs
    """
    annotations = {}
    for idx, det in enumerate(detections, start=1):
        ann_id = f"annotation_{idx:03d}"
        bbox_pixels = normalize_bbox_to_pixels(
            det["bbox"], img_width, img_height
        )
        area = bbox_pixels["width"] * bbox_pixels["height"]
        
        annotations[ann_id] = Annotation(
            category=det["category"],
            bbox=BBox(**bbox_pixels),
            area=area,
            confidence=det.get("confidence")
        )
    return annotations


@app.on_event("startup")
async def startup_event():
    """Initialize detector on startup."""
    print("Initializing detector...")
    get_detector()
    print("Detector ready!")


@app.get("/api")
async def root():
    """API info endpoint."""
    return {"message": "Hybrid Document Detector API", "version": "1.0.0"}


@app.post("/detect")
async def detect_document(
    files: List[UploadFile] = File(...),
):
    """
    Detect signatures, stamps, and QR codes in uploaded PDFs or images (multi-upload).
    
    Returns:
        JSON object in the specified model output format:
        {
          "document.pdf": {
            "page_1": {
              "page_size": {"width": W, "height": H},
              "annotations": [ { "annotation_001": {...}}, ... ]
            }
          }
        }
    """
    try:
        # Generate unique document ID
        doc_id = str(uuid.uuid4())
        doc_dir = STORAGE_DIR / doc_id
        doc_dir.mkdir(parents=True, exist_ok=True)

        detector = get_detector()

        # Combined results across all uploaded files
        combined_results: Dict[str, Dict] = {}

        for upload in files:
            # Save each uploaded file
            upload_path = doc_dir / upload.filename
            with open(upload_path, "wb") as f:
                content = await upload.read()
                f.write(content)

            document_pages: Dict[str, Dict] = {}
            # Continuous annotation counter per document (starts from 1)
            ann_counter: int = 1

            if is_pdf_file(upload.filename):
                print(f"Processing PDF: {upload.filename}")
                pdf_images = pdf_to_images(str(upload_path))

                for page_idx, (img, (width, height)) in enumerate(pdf_images, start=1):
                    page_id = f"page_{page_idx}"

                    # Run detection
                    detections = detector.detect(img)

                    # Convert to annotation format (dict id -> Annotation)
                    ann_dict = process_detections(detections, width, height)

                    # Build annotations array without score; use continuous IDs
                    annotations_array = []
                    for _, ann in ann_dict.items():
                        ann_id_str = f"annotation_{ann_counter:03d}"
                        annotations_array.append({
                            ann_id_str: {
                                "category": ann.category,
                                "bbox": {
                                    "x": ann.bbox.x,
                                    "y": ann.bbox.y,
                                    "width": ann.bbox.width,
                                    "height": ann.bbox.height,
                                },
                                "area": ann.area
                            }
                        })
                        ann_counter += 1

                    # Skip pages with no annotations
                    if len(annotations_array) == 0:
                        continue

                    # Save original and annotated images
                    original_path = doc_dir / f"{page_id}_original.jpg"
                    save_image(img, original_path)
                    annotated_img = detector.annotate_image(img, detections)
                    annotated_path = doc_dir / f"{page_id}_annotated.jpg"
                    save_image(annotated_img, annotated_path)

                    # Store page data
                    document_pages[page_id] = {
                        "page_size": {"width": width, "height": height},
                        "annotations": annotations_array,
                    }

            elif is_image_file(upload.filename):
                print(f"Processing image: {upload.filename}")
                img, (width, height) = read_image(upload_path)

                page_id = "page_1"
                detections = detector.detect(img)
                ann_dict = process_detections(detections, width, height)
                annotations_array = []
                for _, ann in ann_dict.items():
                    ann_id_str = f"annotation_{ann_counter:03d}"
                    annotations_array.append({
                        ann_id_str: {
                            "category": ann.category,
                            "bbox": {
                                "x": ann.bbox.x,
                                "y": ann.bbox.y,
                                "width": ann.bbox.width,
                                "height": ann.bbox.height,
                            },
                            "area": ann.area
                        }
                    })
                    ann_counter += 1

                # Skip page if no annotations
                if len(annotations_array) > 0:
                    original_path = doc_dir / f"{page_id}_original.jpg"
                    save_image(img, original_path)
                    annotated_img = detector.annotate_image(img, detections)
                    annotated_path = doc_dir / f"{page_id}_annotated.jpg"
                    save_image(annotated_img, annotated_path)

                    document_pages[page_id] = {
                        "page_size": {"width": width, "height": height},
                        "annotations": annotations_array,
                    }

            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {upload.filename}. Supported: PDF, JPG, PNG, etc.",
                )

            if document_pages:
                combined_results[upload.filename] = document_pages

        # Persist combined JSON with Unicode preserved
        import json
        json_path = doc_dir / "results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(combined_results, f, indent=2, ensure_ascii=False)

        # Return JSON in required format with document id
        return {
            "document_id": doc_id,
            "results": combined_results,
        }
    
    except Exception as e:
        print(f"Error in detect endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/json/{doc_id}")
async def download_json(doc_id: str):
    """
    Download detection results as JSON file.
    
    Args:
        doc_id: Document ID from detection response
    """
    json_path = STORAGE_DIR / doc_id / "results.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="Document not found or expired")
    
    return FileResponse(
        path=str(json_path),
        filename=f"detections_{doc_id}.json",
        media_type="application/json"
    )


@app.get("/download/zip/{doc_id}")
async def download_zip(doc_id: str):
    """
    Download annotated images as ZIP file.
    
    Args:
        doc_id: Document ID from detection response
    """
    doc_dir = STORAGE_DIR / doc_id
    if not doc_dir.exists():
        raise HTTPException(status_code=404, detail="Document not found or expired")
    
    # Find all annotated images
    annotated_images = sorted(doc_dir.glob("*_annotated.jpg"))
    if not annotated_images:
        raise HTTPException(status_code=404, detail="No annotated images found")
    
    # Create ZIP file
    zip_path = doc_dir / f"annotated_{doc_id}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for img_path in annotated_images:
            zipf.write(img_path, img_path.name)
    
    return FileResponse(
        path=str(zip_path),
        filename=f"annotated_{doc_id}.zip",
        media_type="application/zip"
    )


@app.get("/images/{doc_id}/{page_id}")
async def get_image(doc_id: str, page_id: str, original: bool = False):
    """
    Get image for a specific page (original or annotated).
    
    Args:
        doc_id: Document ID
        page_id: Page ID (e.g., "page_1")
        original: If True, return original image; if False, return annotated
    """
    doc_dir = STORAGE_DIR / doc_id
    if original:
        img_path = doc_dir / f"{page_id}_original.jpg"
    else:
        img_path = doc_dir / f"{page_id}_annotated.jpg"
    
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(
        path=str(img_path),
        media_type="image/jpeg"
    )


# Serve frontend files
# Try multiple paths to find frontend directory
frontend_path = None
possible_paths = [
    Path(__file__).parent.parent / "frontend",  # From backend/main.py -> ../frontend
    Path(__file__).parent / "frontend",  # If running from root
    Path("frontend"),  # Current directory
    Path("../frontend"),  # Relative from backend
]

for path in possible_paths:
    if path.exists() and (path / "index.html").exists():
        frontend_path = path.resolve()
        break

if frontend_path:
    print(f"[Frontend] Serving from: {frontend_path}")
    
    # Define root route BEFORE mounting static files
    @app.get("/", response_class=FileResponse)
    async def serve_frontend():
        """Serve the main frontend page."""
        index_path = frontend_path / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path), media_type="text/html")
        return JSONResponse(
            {"message": "Frontend not found", "path": str(frontend_path)},
            status_code=404
        )
    
    # Mount static files after defining routes
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")
else:
    print("[Frontend] WARNING: Frontend directory not found!")
    print(f"[Frontend] Searched paths: {[str(p) for p in possible_paths]}")
    
    # Fallback route if frontend not found
    @app.get("/")
    async def frontend_not_found():
        return {
            "error": "Frontend not found",
            "searched_paths": [str(p) for p in possible_paths],
            "current_dir": str(Path.cwd())
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)


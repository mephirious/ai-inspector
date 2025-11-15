"""FastAPI backend for Digital Inspector."""

import os
import time
import tempfile
from pathlib import Path
from typing import List
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

try:
    from digital_inspector.detectors import SignatureDetector, QRDetector, StampDetector
    from digital_inspector.utils.pdf_utils import pdf_to_images
    from digital_inspector.utils.image_utils import pil_to_numpy, load_image
    from digital_inspector.utils.merge_utils import merge_detections, format_output
    from digital_inspector.utils.viz_utils import save_annotated_image
except ImportError:
    # Fallback for relative imports
    from ..detectors import SignatureDetector, QRDetector, StampDetector
    from ..utils.pdf_utils import pdf_to_images
    from ..utils.image_utils import pil_to_numpy, load_image
    from ..utils.merge_utils import merge_detections, format_output
    from ..utils.viz_utils import save_annotated_image

from .schemas import DetectionResponseSchema, ErrorResponseSchema

# Initialize FastAPI app
app = FastAPI(
    title="Digital Inspector API",
    description="AI-powered document inspection system for detecting signatures, QR codes, and stamps",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global detector instances (lazy initialization)
signature_detector = None
qr_detector = None
stamp_detector = None

# Output directory
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def get_detectors(device: str = "auto"):
    """Initialize detectors (lazy loading)."""
    global signature_detector, qr_detector, stamp_detector
    
    if signature_detector is None:
        signature_detector = SignatureDetector(device=device)
    if qr_detector is None:
        qr_detector = QRDetector()
    if stamp_detector is None:
        stamp_detector = StampDetector(device=device)
    
    return signature_detector, qr_detector, stamp_detector


def process_image(
    image: np.ndarray,
    page_width: int,
    page_height: int,
    device: str = "auto"
) -> List[dict]:
    """
    Process a single image through all detectors.
    
    Args:
        image: Image as numpy array
        page_width: Original page width
        page_height: Original page height
        device: Device to use for inference
    
    Returns:
        List of merged detections
    """
    sig_det, qr_det, stamp_det = get_detectors(device)
    
    # Run all detectors
    all_detections = []
    
    # Signature detection
    sig_results = sig_det.detect(image)
    all_detections.extend(sig_results)
    
    # QR detection
    qr_results = qr_det.detect(image)
    all_detections.extend(qr_results)
    
    # Stamp detection
    stamp_results = stamp_det.detect(image)
    all_detections.extend(stamp_results)
    
    # Merge and deduplicate
    merged = merge_detections(all_detections, page_width, page_height)
    
    return merged


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Digital Inspector API",
        "version": "1.0.0",
        "endpoints": {
            "/detect": "POST - Upload PDF or image for detection",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Digital Inspector"}


@app.post("/detect", response_model=DetectionResponseSchema)
async def detect(
    file: UploadFile = File(...),
    device: str = "auto",
    save_images: bool = True
):
    """
    Detect signatures, QR codes, and stamps in uploaded file.
    
    Args:
        file: Uploaded PDF or image file
        device: Device to use ('cpu', 'cuda', or 'auto')
        save_images: Whether to save annotated images
    
    Returns:
        Detection results with JSON and preview links
    """
    start_time = time.time()
    
    try:
        # Save uploaded file temporarily
        file_ext = Path(file.filename).suffix.lower()
        is_pdf = file_ext == '.pdf'
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            results = []
            document_name = Path(file.filename).stem
            
            if is_pdf:
                # Process PDF
                pages = pdf_to_images(tmp_path)
                
                for page_image, page_num in pages:
                    image_np = pil_to_numpy(page_image)
                    page_width, page_height = page_image.size
                    
                    # Process page
                    detections = process_image(image_np, page_width, page_height, device)
                    
                    # Format output
                    page_result = format_output(
                        document_name, page_num, page_width, page_height, detections
                    )
                    results.append(page_result)
                    
                    # Save annotated image if requested
                    if save_images:
                        output_subdir = OUTPUT_DIR / document_name
                        output_subdir.mkdir(exist_ok=True)
                        output_path = output_subdir / f"page_{page_num}_annotated.png"
                        save_annotated_image(
                            image_np, str(output_path), detections, page_width, page_height
                        )
            else:
                # Process single image
                image_np = load_image(tmp_path)
                page_height, page_width = image_np.shape[:2]
                
                # Process image
                detections = process_image(image_np, page_width, page_height, device)
                
                # Format output
                page_result = format_output(
                    document_name, 1, page_width, page_height, detections
                )
                results.append(page_result)
                
                # Save annotated image if requested
                if save_images:
                    output_subdir = OUTPUT_DIR / document_name
                    output_subdir.mkdir(exist_ok=True)
                    output_path = output_subdir / "page_1_annotated.png"
                    save_annotated_image(
                        image_np, str(output_path), detections, page_width, page_height
                    )
            
            processing_time = time.time() - start_time
            
            # Generate preview URL (first page if available)
            preview_url = None
            if save_images and results:
                preview_path = OUTPUT_DIR / document_name / "page_1_annotated.png"
                if preview_path.exists():
                    preview_url = f"/preview/{document_name}/page_1_annotated.png"
            
            return DetectionResponseSchema(
                success=True,
                message=f"Processed {len(results)} page(s) successfully",
                results=results,
                preview_url=preview_url,
                processing_time=processing_time
            )
        
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except Exception as e:
        processing_time = time.time() - start_time
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "message": f"Processing failed: {str(e)}",
                "processing_time": processing_time
            }
        )


@app.get("/preview/{document_name}/{filename}")
async def get_preview(document_name: str, filename: str):
    """Serve annotated preview images."""
    preview_path = OUTPUT_DIR / document_name / filename
    if preview_path.exists():
        return FileResponse(str(preview_path))
    raise HTTPException(status_code=404, detail="Preview image not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


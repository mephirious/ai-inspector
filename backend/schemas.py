"""
Pydantic schemas for API request/response models.
"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class BBox(BaseModel):
    """Bounding box in pixel coordinates."""
    x: float = Field(..., description="X coordinate of top-left corner")
    y: float = Field(..., description="Y coordinate of top-left corner")
    width: float = Field(..., description="Width of bounding box")
    height: float = Field(..., description="Height of bounding box")


class PageSize(BaseModel):
    """Page dimensions."""
    width: int = Field(..., description="Page width in pixels")
    height: int = Field(..., description="Page height in pixels")


class Annotation(BaseModel):
    """Single annotation."""
    category: str = Field(..., description="Category: signature, stamp, or qr")
    bbox: BBox = Field(..., description="Bounding box in pixel coordinates")
    area: float = Field(..., description="Area of bounding box (width * height)")
    confidence: Optional[float] = Field(None, description="Detection confidence score")


class PageAnnotations(BaseModel):
    """Annotations for a single page."""
    annotations: Dict[str, Annotation] = Field(..., description="Dictionary of annotations with IDs")
    page_size: PageSize = Field(..., description="Page dimensions")


class DocumentAnnotations(BaseModel):
    """Annotations for all pages in a document."""
    pages: Dict[str, PageAnnotations] = Field(..., description="Dictionary of pages with page IDs")


class DetectionResponse(BaseModel):
    """Response from detection endpoint."""
    document_id: str = Field(..., description="Unique document ID")
    document_name: str = Field(..., description="Original document filename")
    results: Dict[str, Dict[str, PageAnnotations]] = Field(
        ..., 
        description="Nested structure: {document_name: {page_id: PageAnnotations}}"
    )


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")


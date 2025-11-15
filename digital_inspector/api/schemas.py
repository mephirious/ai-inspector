"""Pydantic schemas for API requests and responses."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class DetectionSchema(BaseModel):
    """Schema for a single detection."""
    category: str = Field(..., description="Detection category: signature, qr, or stamp")
    bbox: List[float] = Field(..., description="Bounding box [x, y, width, height] normalized to [0, 1]")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    data: Optional[str] = Field(None, description="Decoded data (for QR codes)")


class PageSizeSchema(BaseModel):
    """Schema for page dimensions."""
    width: int = Field(..., description="Page width in pixels")
    height: int = Field(..., description="Page height in pixels")


class PageResultSchema(BaseModel):
    """Schema for detection results for a single page."""
    document_name: str = Field(..., description="Name of the document")
    page_number: int = Field(..., ge=1, description="Page number (1-indexed)")
    page_size: PageSizeSchema = Field(..., description="Page dimensions")
    detections: List[DetectionSchema] = Field(default_factory=list, description="List of detections")


class DetectionResponseSchema(BaseModel):
    """Schema for API response."""
    success: bool = Field(..., description="Whether processing was successful")
    message: str = Field(..., description="Status message")
    results: List[PageResultSchema] = Field(default_factory=list, description="Detection results per page")
    preview_url: Optional[str] = Field(None, description="URL to preview annotated image")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")


class ErrorResponseSchema(BaseModel):
    """Schema for error responses."""
    success: bool = Field(False, description="Always false for errors")
    message: str = Field(..., description="Error message")
    error: Optional[str] = Field(None, description="Detailed error information")


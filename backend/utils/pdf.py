"""
PDF processing utilities for converting PDF pages to images.
"""
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError, PopplerNotInstalledError


def pdf_to_images(pdf_path: str | Path, dpi: int = 200) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
    """
    Convert PDF pages to OpenCV images with preserved resolution.
    
    Args:
        pdf_path: Path to PDF file
        dpi: DPI for conversion (higher = better quality, larger size)
    
    Returns:
        List of tuples: (image_array, (width, height))
        Each image is a numpy array in BGR format (OpenCV format)
    
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        ValueError: If PDF cannot be processed
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    try:
        # Convert PDF pages to PIL Images
        pil_images = convert_from_path(
            str(pdf_path),
            dpi=dpi,
            fmt='RGB'
        )
        
        if not pil_images:
            raise ValueError(f"No pages found in PDF: {pdf_path}")
        
        # Convert PIL Images to OpenCV format (BGR)
        cv_images = []
        for pil_img in pil_images:
            # Convert PIL RGB to numpy array
            img_array = np.array(pil_img)
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            # Get dimensions
            h, w = img_bgr.shape[:2]
            cv_images.append((img_bgr, (w, h)))
        
        return cv_images
    
    except (PDFInfoNotInstalledError, PopplerNotInstalledError) as e:
        raise ValueError(f"Poppler not installed or not found in PATH: {e}")
    except PDFPageCountError as e:
        raise ValueError(f"Cannot count PDF pages: {e}")
    except Exception as e:
        raise ValueError(f"Error processing PDF: {e}")


def get_pdf_page_count(pdf_path: str | Path) -> int:
    """
    Get the number of pages in a PDF.
    
    Args:
        pdf_path: Path to PDF file
    
    Returns:
        Number of pages
    """
    try:
        pil_images = convert_from_path(str(pdf_path), dpi=72, first_page=1, last_page=1)
        # For full count, we need to try a different approach
        from pdf2image import pdfinfo_from_path
        info = pdfinfo_from_path(str(pdf_path))
        return info.get("Pages", 1)
    except Exception:
        # Fallback: try to convert all pages and count
        try:
            images = convert_from_path(str(pdf_path), dpi=72)
            return len(images)
        except Exception:
            return 1


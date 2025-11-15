"""PDF to image conversion utilities."""

import os
from typing import List, Tuple
from PIL import Image
from pdf2image import convert_from_path
import numpy as np


def pdf_to_images(
    pdf_path: str,
    dpi: int = 350,
    first_page: int = None,
    last_page: int = None
) -> List[Tuple[Image.Image, int]]:
    """
    Convert PDF pages to PIL Images.
    
    Args:
        pdf_path: Path to PDF file
        dpi: Resolution for conversion (300-400 recommended)
        first_page: First page to convert (1-indexed, None for all)
        last_page: Last page to convert (1-indexed, None for all)
    
    Returns:
        List of tuples (PIL Image, page_number)
    """
    try:
        pages = convert_from_path(
            pdf_path,
            dpi=dpi,
            first_page=first_page,
            last_page=last_page,
            fmt='RGB'
        )
        return [(page, idx + 1) for idx, page in enumerate(pages)]
    except Exception as e:
        raise ValueError(f"Failed to convert PDF {pdf_path}: {str(e)}")


def get_pdf_page_count(pdf_path: str) -> int:
    """Get total number of pages in PDF."""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(pdf_path)
        return len(reader.pages)
    except Exception:
        # Fallback: convert first page to get count
        pages = convert_from_path(pdf_path, dpi=100, first_page=1, last_page=1)
        return len(pages) if pages else 0


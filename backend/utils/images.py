"""
Image processing utilities.
"""
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np


def read_image(image_path: str | Path) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Read an image file and return OpenCV format image with dimensions.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Tuple of (image_array, (width, height))
    
    Raises:
        FileNotFoundError: If image cannot be read
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    
    h, w = img.shape[:2]
    return img, (w, h)


def save_image(image: np.ndarray, output_path: str | Path) -> None:
    """
    Save an OpenCV image to file.
    
    Args:
        image: OpenCV image (numpy array)
        output_path: Path where to save the image
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def is_image_file(filename: str) -> bool:
    """
    Check if file is a supported image format.
    
    Args:
        filename: Filename to check
    
    Returns:
        True if file is a supported image format
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    return Path(filename).suffix.lower() in image_extensions


def is_pdf_file(filename: str) -> bool:
    """
    Check if file is a PDF.
    
    Args:
        filename: Filename to check
    
    Returns:
        True if file is a PDF
    """
    return Path(filename).suffix.lower() == '.pdf'


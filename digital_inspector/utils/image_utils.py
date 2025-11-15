"""Image preprocessing and normalization utilities."""

import numpy as np
from PIL import Image
from typing import Union, Tuple
import cv2


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy array (RGB format)."""
    return np.array(image.convert('RGB'))


def numpy_to_pil(image: np.ndarray) -> Image.Image:
    """Convert numpy array to PIL Image."""
    if len(image.shape) == 3 and image.shape[2] == 3:
        return Image.fromarray(image.astype(np.uint8))
    return Image.fromarray(image)


def normalize_image_for_yolo(
    image: np.ndarray,
    target_size: Tuple[int, int] = (640, 640)
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Normalize image for YOLO models while preserving aspect ratio.
    
    Args:
        image: Input image as numpy array (H, W, C)
        target_size: Target size (width, height)
    
    Returns:
        Tuple of (normalized_image, scale_factor, original_size)
    """
    h, w = image.shape[:2]
    original_size = (w, h)
    
    # Calculate scale to fit within target_size while maintaining aspect ratio
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Pad to target_size
    padded = np.full((target_size[1], target_size[0], 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    
    return padded, scale, original_size


def load_image(image_path: str) -> np.ndarray:
    """Load image from file path as numpy array."""
    image = Image.open(image_path)
    return pil_to_numpy(image)


def save_image(image: np.ndarray, output_path: str):
    """Save numpy array as image file."""
    if isinstance(image, np.ndarray):
        image = numpy_to_pil(image)
    image.save(output_path)


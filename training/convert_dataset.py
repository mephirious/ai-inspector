#!/usr/bin/env python3
"""Convert PDFs + JSON annotations to YOLO format dataset."""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
from tqdm import tqdm


# Class mapping
CLASS_MAP = {
    "signature": 0,
    "stamp": 1,
    "qr": 2
}

CLASS_NAMES = ["signature", "stamp", "qr"]


def convert_bbox_to_yolo(
    x: float, y: float, width: float, height: float,
    page_width: int, page_height: int
) -> Tuple[float, float, float, float]:
    """
    Convert bbox from (x, y, width, height) to YOLO format (normalized center, width, height).
    
    Args:
        x: Top-left x coordinate
        y: Top-left y coordinate
        width: Bounding box width
        height: Bounding box height
        page_width: Page width in pixels
        page_height: Page height in pixels
    
    Returns:
        Tuple of (x_center_norm, y_center_norm, width_norm, height_norm)
    """
    x_center = (x + width / 2) / page_width
    y_center = (y + height / 2) / page_height
    w_norm = width / page_width
    h_norm = height / page_height
    
    # Clamp to [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w_norm = max(0.0, min(1.0, w_norm))
    h_norm = max(0.0, min(1.0, h_norm))
    
    return x_center, y_center, w_norm, h_norm


def load_annotations(json_path: str) -> Dict:
    """Load annotations from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def convert_pdf_to_images(
    pdf_path: Path,
    output_dir: Path,
    dpi: int = 350
) -> List[Tuple[Path, int]]:
    """
    Convert PDF to images and return list of (image_path, page_number).
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save images
        dpi: DPI for conversion (300-400 recommended)
    
    Returns:
        List of tuples (image_path, page_number)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_name = pdf_path.stem
    pages = convert_from_path(str(pdf_path), dpi=dpi, fmt='RGB')
    
    image_paths = []
    for idx, page in enumerate(pages, 1):
        image_filename = f"{pdf_name}_page_{idx:04d}.png"
        image_path = output_dir / image_filename
        page.save(image_path, 'PNG')
        image_paths.append((image_path, idx))
    
    return image_paths


def resize_image_to_match(
    image_path: Path,
    target_width: int,
    target_height: int
) -> Image.Image:
    """Resize image to match target dimensions."""
    img = Image.open(image_path)
    if img.size != (target_width, target_height):
        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        img.save(image_path, 'PNG')
    return img


def create_yolo_label(
    annotations: List[Dict],
    page_width: int,
    page_height: int,
    output_path: Path
):
    """
    Create YOLO format label file.
    
    Args:
        annotations: List of annotation dictionaries
        page_width: Page width in pixels
        page_height: Page height in pixels
        output_path: Path to save label file
    """
    label_lines = []
    
    for ann_dict in annotations:
        # Extract annotation data
        for ann_key, ann_data in ann_dict.items():
            category = ann_data.get('category', '').lower()
            bbox = ann_data.get('bbox', {})
            
            if category not in CLASS_MAP:
                continue  # Skip unknown categories
            
            x = bbox.get('x', 0)
            y = bbox.get('y', 0)
            width = bbox.get('width', 0)
            height = bbox.get('height', 0)
            
            # Convert to YOLO format
            x_center, y_center, w_norm, h_norm = convert_bbox_to_yolo(
                x, y, width, height, page_width, page_height
            )
            
            class_id = CLASS_MAP[category]
            label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
    
    # Write label file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(label_lines))


def convert_dataset(
    pdfs_dir: Path,
    annotations_json: Path,
    output_dir: Path,
    dpi: int = 350
):
    """
    Convert PDFs + JSON annotations to YOLO format dataset.
    
    Args:
        pdfs_dir: Directory containing PDF files
        annotations_json: Path to selected_annotations.json
        output_dir: Output directory for dataset
        dpi: DPI for PDF to image conversion
    """
    print("Loading annotations...")
    annotations = load_annotations(annotations_json)
    
    # Create temporary directory for all images
    temp_images_dir = output_dir / "temp_images"
    temp_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each PDF
    pdf_files = list(pdfs_dir.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")
    
    all_samples = []  # List of (image_path, label_path, pdf_name, page_num)
    
    for pdf_path in tqdm(pdf_files, desc="Converting PDFs"):
        pdf_name = pdf_path.name
        
        # Check if PDF has annotations
        if pdf_name not in annotations:
            print(f"Warning: {pdf_name} not found in annotations, skipping")
            continue
        
        # Convert PDF to images
        image_paths = convert_pdf_to_images(pdf_path, temp_images_dir, dpi)
        
        # Process each page
        pdf_annotations = annotations[pdf_name]
        for image_path, page_num in image_paths:
            page_key = f"page_{page_num}"
            
            if page_key not in pdf_annotations:
                # Page has no annotations, skip
                continue
            
            page_data = pdf_annotations[page_key]
            page_size = page_data.get('page_size', {})
            page_width = page_size.get('width', 0)
            page_height = page_size.get('height', 0)
            
            if page_width == 0 or page_height == 0:
                print(f"Warning: Invalid page size for {pdf_name} {page_key}, skipping")
                continue
            
            # Resize image to match page_size if needed
            img = Image.open(image_path)
            if img.size != (page_width, page_height):
                resize_image_to_match(image_path, page_width, page_height)
            
            # Create label file
            label_filename = image_path.stem + '.txt'
            label_path = temp_images_dir / label_filename
            
            page_annotations = page_data.get('annotations', [])
            create_yolo_label(page_annotations, page_width, page_height, label_path)
            
            # Check if label file has content (has annotations)
            if label_path.exists() and label_path.stat().st_size > 0:
                all_samples.append((image_path, label_path, pdf_name, page_num))
            else:
                # No annotations, remove image and label
                image_path.unlink()
                if label_path.exists():
                    label_path.unlink()
    
    print(f"\nTotal samples with annotations: {len(all_samples)}")
    return all_samples


def split_dataset(
    samples: List[Tuple[Path, Path, str, int]],
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List, List]:
    """
    Split dataset into train and validation sets.
    
    Args:
        samples: List of (image_path, label_path, pdf_name, page_num)
        train_ratio: Ratio for training set (default: 0.8)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_samples, val_samples)
    """
    np.random.seed(seed)
    indices = np.random.permutation(len(samples))
    
    split_idx = int(len(samples) * train_ratio)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]
    
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    
    return train_samples, val_samples


def organize_yolo_dataset(
    samples: List[Tuple[Path, Path, str, int]],
    output_dir: Path,
    split_name: str
):
    """
    Organize samples into YOLO dataset structure.
    
    Args:
        samples: List of (image_path, label_path, pdf_name, page_num)
        output_dir: Output directory
        split_name: 'train' or 'val'
    """
    images_dir = output_dir / "images" / split_name
    labels_dir = output_dir / "labels" / split_name
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    for image_path, label_path, pdf_name, page_num in tqdm(samples, desc=f"Organizing {split_name}"):
        # Copy image
        dest_image = images_dir / image_path.name
        shutil.copy2(image_path, dest_image)
        
        # Copy label
        dest_label = labels_dir / label_path.name
        shutil.copy2(label_path, dest_label)


def create_data_yaml(output_dir: Path):
    """Create data.yaml configuration file."""
    data_yaml = f"""# YOLO11 Dataset Configuration
# Generated for Digital Inspector

train: {output_dir.absolute()}/images/train
val: {output_dir.absolute()}/images/val

# Number of classes
nc: 3

# Class names
names:
  0: signature
  1: stamp
  2: qr
"""
    
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(data_yaml)
    
    print(f"Created data.yaml at: {yaml_path}")


def main():
    """Main conversion pipeline."""
    # Paths
    project_root = Path(__file__).parent.parent
    pdfs_dir = project_root / "pdfs"
    annotations_json = project_root / "json" / "selected_annotations.json"
    output_dir = project_root / "training" / "dataset"
    
    # Clean output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("YOLO11 Dataset Conversion Pipeline")
    print("=" * 60)
    print(f"PDFs directory: {pdfs_dir}")
    print(f"Annotations: {annotations_json}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Step 1: Convert PDFs + JSON to YOLO format
    samples = convert_dataset(pdfs_dir, annotations_json, output_dir)
    
    if len(samples) == 0:
        print("Error: No samples found. Check your PDFs and annotations.")
        return
    
    # Step 2: Split dataset
    train_samples, val_samples = split_dataset(samples, train_ratio=0.8)
    
    # Step 3: Organize into YOLO structure
    organize_yolo_dataset(train_samples, output_dir, "train")
    organize_yolo_dataset(val_samples, output_dir, "val")
    
    # Step 4: Create data.yaml
    create_data_yaml(output_dir)
    
    # Cleanup temp directory
    temp_dir = output_dir / "temp_images"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    print("\n" + "=" * 60)
    print("Dataset conversion complete!")
    print(f"Dataset location: {output_dir}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    print("=" * 60)


if __name__ == "__main__":
    main()


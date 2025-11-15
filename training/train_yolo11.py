#!/usr/bin/env python3
"""Train YOLO11s model for Digital Inspector."""

import os
from pathlib import Path
from ultralytics import YOLO
import torch


def main():
    """Main training function."""
    # Paths
    project_root = Path(__file__).parent.parent
    data_yaml = project_root / "training" / "dataset" / "data.yaml"
    model_name = "yolo11s.pt"
    
    # Check if dataset exists
    if not data_yaml.exists():
        print(f"Error: Dataset not found at {data_yaml}")
        print("Please run: python training/convert_dataset.py first")
        return
    
    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cpu":
        print("Warning: Training on CPU will be very slow. Consider using GPU.")
    
    # Load model
    print(f"Loading model: {model_name}")
    model = YOLO(model_name)
    
    # Training hyperparameters (as specified)
    print("\n" + "=" * 60)
    print("Starting YOLO11s Training")
    print("=" * 60)
    print(f"Dataset: {data_yaml}")
    print(f"Device: {device}")
    print("\nHyperparameters:")
    print("  epochs: 150")
    print("  imgsz: 1024")
    print("  batch: -1 (auto)")
    print("  optimizer: AdamW")
    print("  lr0: 0.0015")
    print("  weight_decay: 0.0005")
    print("  momentum: 0.937")
    print("  patience: 20")
    print("  cos_lr: True")
    print("  amp: True (mixed precision)")
    print("=" * 60 + "\n")
    
    # Train model
    results = model.train(
        data=str(data_yaml),
        epochs=150,
        imgsz=1024,
        batch=-1,  # Auto batch size
        optimizer="AdamW",
        lr0=0.0015,
        weight_decay=0.0005,
        momentum=0.937,
        patience=20,
        cos_lr=True,
        device=device,
        amp=True,  # Mixed precision
        # Augmentations (document-optimized)
        mosaic=1.0,
        hsv_h=0.2,
        hsv_s=0.2,
        hsv_v=0.2,
        fliplr=0.5,
        flipud=0.0,
        degrees=0.0,
        perspective=0.0,
        mixup=0.0,
        # Project name
        project=str(project_root / "training" / "runs"),
        name="yolo11_finetuned",
        exist_ok=True,
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Best model: {model.trainer.best}")
    print(f"Results saved to: {model.trainer.save_dir}")
    print("=" * 60)
    
    # Export to ONNX
    print("\nExporting to ONNX...")
    try:
        model.export(format="onnx", opset=12, simplify=True)
        print(f"ONNX model exported: {model.trainer.save_dir / 'best.onnx'}")
    except Exception as e:
        print(f"Warning: ONNX export failed: {e}")
    
    print("\nTraining pipeline complete!")


if __name__ == "__main__":
    main()


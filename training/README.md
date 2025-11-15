# 🎯 YOLO11 Fine-Tuning Guide for Digital Inspector

Complete guide for training YOLO11s on construction document dataset.

---

## 📋 Overview

This training pipeline fine-tunes **YOLO11s** to detect:
- **0: signature** - Handwritten and digital signatures
- **1: stamp** - Official stamps and seals
- **2: qr** - QR codes

Using 45 construction PDFs with JSON annotations.

---

## 🚀 Quick Start

### Step 1: Convert Dataset

```bash
python training/convert_dataset.py
```

This will:
- Convert PDFs to PNG images (300-400 DPI)
- Parse `selected_annotations.json`
- Convert bboxes to YOLO format
- Create train/val split (80/20)
- Generate `dataset/data.yaml`

**Output:**
```
training/dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
  data.yaml
```

### Step 2: Train Model

```bash
python training/train_yolo11.py
```

**Training Parameters:**
- Model: YOLO11s
- Epochs: 150
- Image size: 1024x1024
- Batch: Auto (GPU)
- Optimizer: AdamW
- Learning rate: 0.0015
- Mixed precision: Enabled

**Output:**
```
training/runs/yolo11_finetuned/
  weights/
    best.pt      # Best model
    last.pt      # Last checkpoint
    best.onnx    # ONNX export
  results.csv
  charts.png
```

### Step 3: Use Fine-Tuned Model

```bash
# Use fine-tuned model
python run.py --pdf pdfs/отр-22.pdf --output results/ --model training/runs/yolo11_finetuned/weights/best.pt
```

---

## 📊 Dataset Statistics

After conversion, you should see:
- Total samples with annotations
- Train/val split counts
- Class distribution

---

## 🔧 Advanced Configuration

### Custom Model Path

```python
from digital_inspector.detectors.yolo11_detector import YOLO11Detector

detector = YOLO11Detector(
    model_path="path/to/custom/model.pt",
    device="cuda",
    confidence_threshold=0.25
)
```

### Training Custom Hyperparameters

Edit `training/train_yolo11.py` to modify:
- Epochs
- Image size
- Learning rate
- Augmentations

---

## 📈 Monitoring Training

Training progress is saved to:
- `results.csv` - Metrics per epoch
- `charts.png` - Visualization
- TensorBoard logs (if enabled)

Monitor:
- mAP50 (mean Average Precision)
- mAP50-95
- Precision/Recall
- Loss curves

---

## 🎯 Expected Results

After 150 epochs, you should achieve:
- **mAP50**: > 0.85 (good)
- **mAP50-95**: > 0.60 (good)
- **Precision**: > 0.90
- **Recall**: > 0.85

---

## 🐛 Troubleshooting

### Out of Memory

Reduce batch size or image size:
```python
batch=8  # Instead of -1
imgsz=640  # Instead of 1024
```

### Slow Training

- Ensure GPU is being used (`device="cuda"`)
- Enable mixed precision (`amp=True`)
- Reduce image size if needed

### No Annotations Found

- Check `selected_annotations.json` format
- Verify PDF names match JSON keys
- Ensure page numbers match

---

## 📚 Files

- `convert_dataset.py` - Dataset conversion script
- `train_yolo11.py` - Training script
- `dataset/` - Generated dataset
- `runs/` - Training outputs

---

## 🔄 Integration

The fine-tuned model integrates seamlessly with Digital Inspector:

```python
from digital_inspector.detectors.yolo11_detector import YOLO11Detector

# Unified detector for all classes
detector = YOLO11Detector(model_path="best.pt")
detections = detector.detect(image)
```

---

## 📝 Notes

- Training on GPU is **highly recommended** (10-20x faster)
- First epoch may be slow (model initialization)
- Best model is saved automatically based on validation mAP
- ONNX export enables faster inference

---

**Happy Training! 🚀**


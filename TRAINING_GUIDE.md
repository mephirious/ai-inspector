# 🎯 Complete YOLO11 Training Pipeline - Digital Inspector

## ✅ What's Been Generated

I've created a **complete YOLO11 fine-tuning pipeline** for Digital Inspector. Here's what's ready:

### 📁 Files Created

1. **`training/convert_dataset.py`** - Converts PDFs + JSON → YOLO format
2. **`training/train_yolo11.py`** - Training script with all hyperparameters
3. **`training/README.md`** - Detailed training guide
4. **`digital_inspector/detectors/yolo11_detector.py`** - Unified YOLO11 detector
5. **Updated `run.py`** - CLI now supports YOLO11 models
6. **Updated `requirements.txt`** - Added tqdm for progress bars
7. **Updated main `README.md`** - Added training section

---

## 🚀 Quick Start (3 Steps)

### Step 1: Convert Dataset

```bash
python training/convert_dataset.py
```

**What it does:**
- Reads all PDFs from `pdfs/`
- Parses `json/selected_annotations.json`
- Converts PDFs to PNG images (350 DPI)
- Converts bboxes to YOLO format (normalized center, width, height)
- Creates 80/20 train/val split
- Generates `dataset/data.yaml`

**Output:**
```
training/dataset/
  images/
    train/  (80% of samples)
    val/    (20% of samples)
  labels/
    train/
    val/
  data.yaml
```

### Step 2: Train YOLO11s

```bash
python training/train_yolo11.py
```

**Training Configuration:**
- Model: YOLO11s
- Epochs: 150
- Image size: 1024x1024
- Batch: Auto (GPU)
- Optimizer: AdamW
- Learning rate: 0.0015
- Mixed precision: Enabled
- Document-optimized augmentations

**Output:**
```
training/runs/yolo11_finetuned/
  weights/
    best.pt      # Best model (use this!)
    last.pt      # Last checkpoint
    best.onnx    # ONNX export (faster inference)
  results.csv
  charts.png
```

### Step 3: Use Fine-Tuned Model

```bash
# Use your fine-tuned model
python run.py --pdf pdfs/отр-22.pdf --output results/ \
  --model training/runs/yolo11_finetuned/weights/best.pt
```

---

## 📊 Dataset Details

### Classes
- **0: signature** - Handwritten and digital signatures
- **1: stamp** - Official stamps and seals  
- **2: qr** - QR codes

### Annotation Format

Your `selected_annotations.json` format:
```json
{
  "локалсмета-.pdf": {
    "page_3": {
      "page_size": { "width": 1684, "height": 1190 },
      "annotations": [
        {
          "annotation_117": {
            "category": "signature",
            "bbox": { "x": 510, "y": 146, "width": 250, "height": 98.89 }
          }
        }
      ]
    }
  }
}
```

### Conversion Process

1. **PDF → Image**: 350 DPI PNG conversion
2. **Bbox Conversion**: `(x, y, width, height)` → YOLO `(x_center, y_center, width, height)` (normalized)
3. **Image Resizing**: Matches `page_size` from JSON
4. **Filtering**: Only pages with annotations are included

---

## 🎯 Training Hyperparameters

All specified in `train_yolo11.py`:

```python
epochs=150
imgsz=1024
batch=-1              # Auto GPU batch
optimizer="AdamW"
lr0=0.0015
weight_decay=0.0005
momentum=0.937
patience=20
cos_lr=True
device="cuda"
amp=True              # Mixed precision

# Augmentations
mosaic=1.0
hsv_h=0.2
hsv_s=0.2
hsv_v=0.2
fliplr=0.5
flipud=0.0
degrees=0.0
perspective=0.0
mixup=0.0
```

---

## 🔧 Integration

### Using YOLO11 Detector in Code

```python
from digital_inspector.detectors.yolo11_detector import YOLO11Detector

# Load fine-tuned model
detector = YOLO11Detector(
    model_path="training/runs/yolo11_finetuned/weights/best.pt",
    device="cuda",
    confidence_threshold=0.25
)

# Detect all classes at once
detections = detector.detect(image)
# Returns: [{"category": "signature", "bbox": [...], "confidence": 0.95}, ...]
```

### CLI Usage

```bash
# Use fine-tuned YOLO11 model
python run.py --pdf document.pdf --output results/ \
  --model training/runs/yolo11_finetuned/weights/best.pt

# Use separate detectors (default)
python run.py --pdf document.pdf --output results/
```

---

## 📈 Expected Results

After 150 epochs on GPU:

- **Training time**: ~2-4 hours (depending on GPU)
- **mAP50**: > 0.85 (good)
- **mAP50-95**: > 0.60 (good)
- **Precision**: > 0.90
- **Recall**: > 0.85

---

## 🐛 Troubleshooting

### Out of Memory

Edit `training/train_yolo11.py`:
```python
batch=8  # Instead of -1
imgsz=640  # Instead of 1024
```

### No Annotations Found

- Check PDF names match JSON keys exactly
- Verify page numbers match (page_1, page_2, etc.)
- Ensure categories are: "signature", "stamp", or "qr"

### Slow Training

- Ensure GPU is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Check GPU usage: `nvidia-smi`
- Reduce image size if needed

---

## 📚 File Structure

```
ai-inspector/
├── training/
│   ├── convert_dataset.py    # Dataset converter
│   ├── train_yolo11.py       # Training script
│   ├── README.md              # Training guide
│   ├── dataset/               # Generated dataset
│   │   ├── images/
│   │   ├── labels/
│   │   └── data.yaml
│   └── runs/                  # Training outputs
│       └── yolo11_finetuned/
│           └── weights/
│               ├── best.pt
│               └── best.onnx
├── digital_inspector/
│   └── detectors/
│       └── yolo11_detector.py  # Unified detector
└── run.py                      # Updated CLI
```

---

## ✅ Next Steps

1. **Run dataset conversion**: `python training/convert_dataset.py`
2. **Check dataset stats**: Review output for sample counts
3. **Start training**: `python training/train_yolo11.py`
4. **Monitor progress**: Check `training/runs/yolo11_finetuned/results.csv`
5. **Use best model**: Point `--model` to `best.pt`

---

## 🎉 Ready to Train!

Everything is set up and ready. Just run the two scripts and you'll have a fine-tuned YOLO11s model optimized for your construction documents!

**Happy Training! 🚀**


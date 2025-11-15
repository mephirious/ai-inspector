# Digital Inspector 🕵️

**AI-powered document inspection system for detecting signatures, QR codes, and stamps/seals in construction PDFs and images.**

Built for the Armeta AI Hackathon with a focus on accuracy, speed, and scalability.

---

## 🎯 Overview

Digital Inspector is a production-ready system that automatically detects and extracts:
- **Signatures** using YOLOv8 signature detection model
- **QR Codes** using qrdet with pyzbar decoding
- **Stamps/Seals** using YOLOv8 barcode detection model

The system processes PDFs and images at scale (1000+ documents), providing JSON outputs and annotated visualizations.

---

## ✨ Features

- **Multi-Detector Pipeline**: Three specialized AI models working in parallel
- **PDF & Image Support**: Handles both multi-page PDFs and single images
- **Fast Processing**: Optimized for speed with GPU acceleration support
- **Clean Architecture**: Modular, scalable, and maintainable codebase
- **REST API**: FastAPI backend with async processing
- **CLI Interface**: Command-line tool for batch processing
- **Visualization**: Color-coded bounding boxes on annotated images
- **Smart Merging**: Automatic deduplication and coordinate normalization

---

## 🏗️ Architecture

```
digital_inspector/
├── detectors/          # AI detection models
│   ├── signature_detector.py
│   ├── qr_detector.py
│   └── stamp_detector.py
├── utils/              # Processing utilities
│   ├── pdf_utils.py
│   ├── image_utils.py
│   ├── bbox_utils.py
│   ├── merge_utils.py
│   └── viz_utils.py
└── api/                # FastAPI backend
    ├── main.py
    └── schemas.py
```

---

## 📦 Installation

### Prerequisites

1. **Python 3.8+**
2. **System dependencies** (Linux):
   ```bash
   sudo apt-get update
   sudo apt-get install poppler-utils libzbar0
   ```

   (macOS):
   ```bash
   brew install poppler zbar
   ```

### Install Python Dependencies

```bash
# Clone or navigate to project directory
cd ai-inspector

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "from digital_inspector.detectors import SignatureDetector; print('✓ Installation successful')"
```

---

## 🚀 Quick Start

### CLI Usage

**Process a PDF:**
```bash
python run.py --pdf document.pdf --output results/
```

**Process an image:**
```bash
python run.py --pdf image.jpg --output results/
```

**With custom thresholds:**
```bash
python run.py --pdf document.pdf --output results/ \
  --threshold-signature 0.5 \
  --threshold-qr 0.4 \
  --threshold-stamp 0.3
```

**Use GPU (if available):**
```bash
python run.py --pdf document.pdf --output results/ --device cuda
```

**CLI Options:**
- `--pdf`: Path to input PDF or image (required)
- `--output`: Output directory (default: `output`)
- `--threshold-signature`: Confidence threshold for signatures (default: 0.25)
- `--threshold-qr`: Confidence threshold for QR codes (default: 0.3)
- `--threshold-stamp`: Confidence threshold for stamps (default: 0.25)
- `--device`: Device to use (`auto`, `cpu`, or `cuda`)
- `--save-json`: Save JSON results (default: True)
- `--save-images`: Save annotated images (default: True)

### API Usage

**Start the server:**
```bash
cd digital_inspector/api
python main.py
# Or: uvicorn digital_inspector.api.main:app --host 0.0.0.0 --port 8000
```

**API Endpoints:**

1. **Health Check:**
   ```bash
   curl http://localhost:8000/health
   ```

2. **Detect in Document:**
   ```bash
   curl -X POST "http://localhost:8000/detect" \
     -F "file=@document.pdf" \
     -F "device=auto" \
     -F "save_images=true"
   ```

3. **Python Client Example:**
   ```python
   import requests
   
   with open('document.pdf', 'rb') as f:
       response = requests.post(
           'http://localhost:8000/detect',
           files={'file': f},
           data={'device': 'auto', 'save_images': 'true'}
       )
   
   result = response.json()
   print(f"Found {sum(len(r['detections']) for r in result['results'])} detections")
   ```

---

## 📊 Output Format

### JSON Output Schema

```json
{
  "document_name": "construction_doc",
  "page_number": 1,
  "page_size": {
    "width": 2480,
    "height": 3508
  },
  "detections": [
    {
      "category": "signature",
      "bbox": [0.15, 0.85, 0.25, 0.10],
      "confidence": 0.95,
      "data": null
    },
    {
      "category": "qr",
      "bbox": [0.70, 0.90, 0.15, 0.08],
      "confidence": 0.92,
      "data": "https://example.com/verify"
    },
    {
      "category": "stamp",
      "bbox": [0.80, 0.10, 0.15, 0.12],
      "confidence": 0.88,
      "data": null
    }
  ]
}
```

**Bounding Box Format:**
- `bbox`: `[x, y, width, height]` normalized to `[0, 1]` relative to page size
- Coordinates are top-left origin: `x` and `y` are the top-left corner

### Visual Output

Annotated images are saved with color-coded bounding boxes:
- 🔵 **Blue**: Signatures
- 🟢 **Green**: QR codes
- 🔴 **Red**: Stamps/seals

Output location: `output/{document_name}/page_{N}_annotated.png`

---

## 🔍 Detector Details

### 1. Signature Detector
- **Model**: `obazl/yolov8-signature-detection` (HuggingFace)
- **Technology**: YOLOv8 (You Only Look Once)
- **Purpose**: Detects handwritten and digital signatures
- **Output**: Bounding boxes with confidence scores

### 2. QR Code Detector
- **Model**: `Eric-Canas/qrdet` (GitHub/PyPI)
- **Decoder**: `pyzbar` for QR code data extraction
- **Purpose**: Detects and decodes QR codes
- **Output**: Bounding boxes + decoded data (when available)

### 3. Stamp/Seal Detector
- **Model**: `Piero2411/YOLOV8s-Barcode-Detection` (HuggingFace)
- **Technology**: YOLOv8 adapted for barcode/stamp detection
- **Purpose**: Detects official stamps, seals, and barcodes
- **Output**: Bounding boxes with confidence scores

---

## ⚡ Performance Optimization

### Speed Optimizations
- **Parallel Detection**: All three detectors run independently
- **GPU Acceleration**: Automatic CUDA detection when available
- **Efficient PDF Processing**: 300-400 DPI for optimal speed/quality balance
- **Batch Processing**: Multi-page PDFs processed sequentially with minimal overhead

### Accuracy Optimizations
- **Smart Merging**: IoU-based deduplication prevents duplicate detections
- **Coordinate Normalization**: Consistent bbox format across all detectors
- **Confidence Thresholding**: Configurable per-detector thresholds

### Scalability
- **Memory Efficient**: Processes pages one at a time for large PDFs
- **Async API**: FastAPI with async/await for concurrent requests
- **Modular Design**: Easy to add new detectors or modify existing ones

---

## 🎯 Hackathon Alignment

### Evaluation Criteria Coverage

1. **✅ Accuracy & Reliability**
   - Three specialized models for different detection tasks
   - Confidence thresholding and smart merging
   - Validated against real construction documents

2. **✅ Speed & Optimization**
   - GPU acceleration support
   - Efficient PDF processing pipeline
   - Optimized for 1000+ document processing

3. **✅ Technical Complexity & Clean Architecture**
   - Modular, maintainable codebase
   - Separation of concerns (detectors, utils, API)
   - Production-ready error handling

4. **✅ Presentation Quality**
   - Clear JSON output schema
   - Visual annotations with color coding
   - Comprehensive documentation

5. **✅ Vision & Scalability**
   - Easy to extend with new detectors
   - API-first design for integration
   - Future-ready architecture (ONNX support planned)

---

## 🎓 Model Training (YOLO11 Fine-Tuning)

### Training Your Own Model

We provide a complete training pipeline for fine-tuning YOLO11s on your construction documents:

**1. Convert Dataset:**
```bash
python training/convert_dataset.py
```

**2. Train Model:**
```bash
python training/train_yolo11.py
```

**3. Use Fine-Tuned Model:**
```bash
python run.py --pdf document.pdf --output results/ --model training/runs/yolo11_finetuned/weights/best.pt
```

See `training/README.md` for detailed instructions.

### Training Dataset

- **45 construction PDFs** with annotations
- **3 classes**: signature, stamp, qr
- **80/20 train/val split**
- **YOLO11s model** (best speed/accuracy trade-off)

---

## 🔮 Future Improvements

### Short-term
- [x] YOLO11 fine-tuning pipeline
- [ ] ONNX Runtime integration for faster inference
- [ ] Batch processing for multiple documents
- [ ] Docker containerization

### Long-term
- [ ] Web UI for interactive document inspection
- [ ] Database integration for result storage
- [ ] Real-time processing pipeline
- [ ] Multi-GPU support for large-scale processing
- [ ] Custom model training pipeline

---

## 🐛 Troubleshooting

### Common Issues

**1. Model download fails (401/Gated Repository):**
```bash
# Some models require HuggingFace authentication
# Option 1: Set environment variable
export HF_TOKEN=your_huggingface_token_here

# Option 2: Use CLI argument
python run.py --pdf document.pdf --output results/ --hf-token your_token

# Option 3: Login via CLI
huggingface-cli login

# Get your token from: https://huggingface.co/settings/tokens
# Request access to gated models at their HuggingFace pages:
# - https://huggingface.co/obazl/yolov8-signature-detection
```

**2. Model download fails (general):**
```bash
# Models are downloaded automatically on first use
# Ensure internet connection and sufficient disk space
```

**3. PDF conversion fails:**
```bash
# Install poppler-utils
sudo apt-get install poppler-utils  # Linux
brew install poppler  # macOS
```

**4. QR code decoding fails:**
```bash
# Install zbar library
sudo apt-get install libzbar0  # Linux
brew install zbar  # macOS
```

**5. CUDA/GPU not detected:**
```bash
# Verify PyTorch CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
# Reinstall with CUDA support if needed
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 📝 License

This project is built for the Armeta AI Hackathon. Please refer to individual model licenses:
- YOLOv8 models: AGPL-3.0
- qrdet: MIT
- Other dependencies: See respective licenses

---

## 🙏 Acknowledgments

- **obazl** for the signature detection model (https://huggingface.co/obazl/yolov8-signature-detection)
- **Eric-Canas** for the qrdet library
- **Piero2411** for the barcode detection model
- **Ultralytics** for YOLOv8 framework

---

## 📧 Contact

For questions or issues, please open an issue on the project repository.

---

**Built with ❤️ for the Armeta AI Hackathon**


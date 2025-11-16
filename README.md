# Hybrid Document Detector Web Service

A production-ready web application for detecting signatures, stamps, and QR codes in PDFs and images using a 3-model hybrid YOLO detector.

## Features

- **Web Interface**: Upload PDFs or images with drag-and-drop support
- **Real-time Detection**: Process documents with 3 YOLO models (signature, stamp, QR)
- **Interactive Preview**: View annotated pages with bounding boxes
- **Category Toggles**: Show/hide specific detection categories
- **Export Options**: Download results as JSON or annotated images (ZIP)
- **RESTful API**: Full FastAPI backend with proper endpoints

## Project Structure

```
.
├── backend/
│   ├── detector/
│   │   └── hybrid_detector.py    # Core detection logic (3-model hybrid)
│   ├── utils/
│   │   ├── pdf.py                # PDF to image conversion
│   │   └── images.py             # Image processing utilities
│   ├── schemas.py                # Pydantic models for API
│   └── main.py                   # FastAPI application
├── frontend/
│   ├── index.html                # Main HTML page
│   ├── app.js                    # Frontend JavaScript logic
│   └── styles.css                # Modern CSS styling
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (optional, for faster inference)

### Setup

1. **Clone or navigate to the project directory**

2. **Create a virtual environment** (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

4. **Ensure model files are in place**:
   - `qr.pt` - QR code detection model
   - `stamp.pt` - Stamp detection model
   - `signature_best.pt` - Signature detection model

   Update paths in `backend/detector/hybrid_detector.py` if models are located elsewhere.

## Usage

### Starting the Server

```bash
cd backend
python main.py
```

Or using uvicorn directly:
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Or from the project root:
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

The web interface will be available at: `http://localhost:8000`

### API Endpoints

#### POST `/detect`
Upload a PDF or image file for detection.

**Request:**
- `file`: PDF or image file (multipart/form-data)

**Response:**
```json
{
  "document_id": "uuid-string",
  "document_name": "document.pdf",
  "results": {
    "document.pdf": {
      "page_1": {
        "annotations": {
          "annotation_001": {
            "category": "signature",
            "bbox": {"x": 510, "y": 146, "width": 250, "height": 98.89},
            "area": 24722.5,
            "confidence": 0.95
          }
        },
        "page_size": {"width": 1684, "height": 1190}
      }
    }
  }
}
```

#### GET `/download/json/{doc_id}`
Download detection results as JSON file.

#### GET `/download/zip/{doc_id}`
Download annotated images as ZIP file.

### Using the Web Interface

1. **Upload**: Click the upload area or drag-and-drop a PDF/image file
2. **Process**: Click "Process Document" button
3. **View Results**: 
   - See annotated pages with bounding boxes
   - Toggle categories (signature/stamp/QR) to show/hide
   - View detection details in the annotations list
4. **Download**: 
   - Click "Download JSON" for structured results
   - Click "Download ZIP" for annotated images

## Detection Logic

The system uses a hybrid approach with 3 YOLO models:

1. **QR Model** (`qr.pt`): Detects QR codes at 640px
2. **Stamp Model** (`stamp.pt`): Detects stamps (class 15) at 640px
3. **Signature Model** (`signature_best.pt`): Detects signatures at 640px and 1024px

All detections are merged using IoU threshold (0.5) to eliminate duplicates, keeping the highest confidence detection within each category.

## Configuration

Detection parameters can be adjusted in `backend/detector/hybrid_detector.py`:

- `QR_CONF = 0.5` - QR detection confidence threshold
- `STAMP_CONF = 0.1` - Stamp detection confidence threshold
- `SIGNATURE_CONF = 0.1` - Signature detection confidence threshold
- `MERGE_IOU = 0.5` - IoU threshold for merging detections

## Storage

Temporary files are stored in `/tmp/detector_storage/{document_id}/`:
- Original uploaded file
- Annotated images (per page)
- Results JSON

Files are kept for download but can be cleaned up periodically.

## Development

### Code Structure

- **Modular Design**: Detector logic is separated from API and utilities
- **Type Hints**: Full type annotations throughout
- **Error Handling**: Comprehensive error handling with proper HTTP status codes
- **Pydantic Models**: Type-safe API schemas

### Adding Features

- New detection categories: Modify `HybridDetector3Models` class
- Additional file formats: Extend `utils/images.py` and `utils/pdf.py`
- Custom endpoints: Add routes in `backend/main.py`

## Troubleshooting

### PDF Processing Issues
- Ensure `poppler` is installed and in PATH
- Check PDF file is not corrupted or password-protected

### Model Loading Errors
- Verify model file paths in `hybrid_detector.py`
- Ensure models are compatible with ultralytics version

### GPU Not Detected
- Install CUDA-compatible PyTorch version
- Check CUDA installation: `nvidia-smi`

## License

This project maintains the original detection logic and wraps it in a production-ready web service.

## Support

For issues or questions, please check:
- Model compatibility with ultralytics
- System dependencies (poppler, CUDA)
- File format support


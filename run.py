#!/usr/bin/env python3
"""CLI interface for Digital Inspector."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

from digital_inspector.detectors import (
    SignatureDetector, QRDetector, StampDetector,
    YOLO11Detector
)
from digital_inspector.utils.pdf_utils import pdf_to_images
from digital_inspector.utils.image_utils import pil_to_numpy, load_image
from digital_inspector.utils.merge_utils import merge_detections, format_output
from digital_inspector.utils.viz_utils import save_annotated_image


def process_document(
    input_path: str,
    output_dir: str,
    threshold_signature: float = 0.25,
    threshold_qr: float = 0.3,
    threshold_stamp: float = 0.25,
    device: str = "auto",
    save_json: bool = True,
    save_images: bool = True,
    hf_token: str = None,
    model_path: str = None
):
    """
    Process a document through the detection pipeline.
    
    Args:
        input_path: Path to input PDF or image
        output_dir: Directory to save outputs
        threshold_signature: Confidence threshold for signatures
        threshold_qr: Confidence threshold for QR codes
        threshold_stamp: Confidence threshold for stamps
        device: Device to use ('cpu', 'cuda', or 'auto')
        save_json: Whether to save JSON results
        save_images: Whether to save annotated images
    """
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize detectors
    print("Initializing detectors...")
    use_yolo11 = model_path is not None
    
    if use_yolo11:
        # Use unified YOLO11 detector
        try:
            yolo11_detector = YOLO11Detector(
                model_path=model_path,
                device=device,
                confidence_threshold=min(threshold_signature, threshold_qr, threshold_stamp)
            )
            print("YOLO11 detector initialized successfully.")
        except Exception as e:
            print(f"Error initializing YOLO11 detector: {str(e)}")
            sys.exit(1)
    else:
        # Use separate detectors
        try:
            signature_detector = SignatureDetector(
                device=device,
                confidence_threshold=threshold_signature,
                token=hf_token
            )
            qr_detector = QRDetector(confidence_threshold=threshold_qr)
            stamp_detector = StampDetector(
                device=device,
                confidence_threshold=threshold_stamp,
                token=hf_token
            )
            print("Detectors initialized successfully.")
        except Exception as e:
            print(f"Error initializing detectors: {str(e)}")
            sys.exit(1)
    
    # Determine file type
    is_pdf = input_path.suffix.lower() == '.pdf'
    document_name = input_path.stem
    
    # Create output subdirectory
    doc_output_dir = output_dir / document_name
    doc_output_dir.mkdir(exist_ok=True)
    
    all_results = []
    start_time = time.time()
    
    try:
        if is_pdf:
            print(f"Processing PDF: {input_path}")
            pages = pdf_to_images(str(input_path))
            print(f"Found {len(pages)} page(s)")
            
            for page_idx, (page_image, page_num) in enumerate(pages, 1):
                print(f"Processing page {page_num}/{len(pages)}...")
                image_np = pil_to_numpy(page_image)
                page_width, page_height = page_image.size
                
                # Run detectors
                all_detections = []
                
                if use_yolo11:
                    # Use unified YOLO11 detector
                    print("  Running YOLO11 detection...")
                    yolo11_results = yolo11_detector.detect(image_np)
                    all_detections.extend(yolo11_results)
                    print(f"    Found {len(yolo11_results)} detection(s)")
                else:
                    # Use separate detectors
                    # Signature detection
                    print("  Detecting signatures...")
                    sig_results = signature_detector.detect(image_np)
                    all_detections.extend(sig_results)
                    print(f"    Found {len(sig_results)} signature(s)")
                    
                    # QR detection
                    print("  Detecting QR codes...")
                    qr_results = qr_detector.detect(image_np)
                    all_detections.extend(qr_results)
                    print(f"    Found {len(qr_results)} QR code(s)")
                    
                    # Stamp detection
                    print("  Detecting stamps/seals...")
                    stamp_results = stamp_detector.detect(image_np)
                    all_detections.extend(stamp_results)
                    print(f"    Found {len(stamp_results)} stamp(s)")
                
                # Merge detections
                print("  Merging detections...")
                merged = merge_detections(all_detections, page_width, page_height)
                print(f"    Total unique detections: {len(merged)}")
                
                # Format output
                page_result = format_output(
                    document_name, page_num, page_width, page_height, merged
                )
                all_results.append(page_result)
                
                # Save annotated image
                if save_images:
                    output_path = doc_output_dir / f"page_{page_num}_annotated.png"
                    save_annotated_image(
                        image_np, str(output_path), merged, page_width, page_height
                    )
                    print(f"    Saved annotated image: {output_path}")
        
        else:
            print(f"Processing image: {input_path}")
            image_np = load_image(str(input_path))
            page_height, page_width = image_np.shape[:2]
            
            # Run detectors
            all_detections = []
            
            if use_yolo11:
                # Use unified YOLO11 detector
                print("  Running YOLO11 detection...")
                yolo11_results = yolo11_detector.detect(image_np)
                all_detections.extend(yolo11_results)
                print(f"    Found {len(yolo11_results)} detection(s)")
            else:
                # Use separate detectors
                print("  Detecting signatures...")
                sig_results = signature_detector.detect(image_np)
                all_detections.extend(sig_results)
                print(f"    Found {len(sig_results)} signature(s)")
                
                print("  Detecting QR codes...")
                qr_results = qr_detector.detect(image_np)
                all_detections.extend(qr_results)
                print(f"    Found {len(qr_results)} QR code(s)")
                
                print("  Detecting stamps/seals...")
                stamp_results = stamp_detector.detect(image_np)
                all_detections.extend(stamp_results)
                print(f"    Found {len(stamp_results)} stamp(s)")
            
            # Merge detections
            print("  Merging detections...")
            merged = merge_detections(all_detections, page_width, page_height)
            print(f"    Total unique detections: {len(merged)}")
            
            # Format output
            page_result = format_output(
                document_name, 1, page_width, page_height, merged
            )
            all_results.append(page_result)
            
            # Save annotated image
            if save_images:
                output_path = doc_output_dir / "page_1_annotated.png"
                save_annotated_image(
                    image_np, str(output_path), merged, page_width, page_height
                )
                print(f"    Saved annotated image: {output_path}")
        
        # Save JSON results
        if save_json:
            json_path = doc_output_dir / "results.json"
            with open(json_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\nSaved JSON results: {json_path}")
        
        processing_time = time.time() - start_time
        total_detections = sum(len(r['detections']) for r in all_results)
        
        print("\n" + "="*50)
        print("Processing Summary")
        print("="*50)
        print(f"Document: {document_name}")
        print(f"Pages processed: {len(all_results)}")
        print(f"Total detections: {total_detections}")
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Output directory: {doc_output_dir}")
        print("="*50)
    
    except Exception as e:
        print(f"\nError processing document: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Digital Inspector - AI-powered document inspection system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a PDF
  python run.py --pdf document.pdf --output results/
  
  # Process an image with custom thresholds
  python run.py --pdf image.jpg --output results/ --threshold-signature 0.5
  
  # Process with GPU
  python run.py --pdf document.pdf --output results/ --device cuda
        """
    )
    
    parser.add_argument(
        "--pdf",
        type=str,
        required=True,
        help="Path to input PDF or image file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory (default: output)"
    )
    
    parser.add_argument(
        "--threshold-signature",
        type=float,
        default=0.25,
        help="Confidence threshold for signature detection (default: 0.25)"
    )
    
    parser.add_argument(
        "--threshold-qr",
        type=float,
        default=0.3,
        help="Confidence threshold for QR code detection (default: 0.3)"
    )
    
    parser.add_argument(
        "--threshold-stamp",
        type=float,
        default=0.25,
        help="Confidence threshold for stamp detection (default: 0.25)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for inference (default: auto)"
    )
    
    parser.add_argument(
        "--save-json",
        action="store_true",
        default=True,
        help="Save JSON results (default: True)"
    )
    
    parser.add_argument(
        "--no-save-json",
        dest="save_json",
        action="store_false",
        help="Don't save JSON results"
    )
    
    parser.add_argument(
        "--save-images",
        action="store_true",
        default=True,
        help="Save annotated images (default: True)"
    )
    
    parser.add_argument(
        "--no-save-images",
        dest="save_images",
        action="store_false",
        help="Don't save annotated images"
    )
    
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token for gated repositories (or set HF_TOKEN env var)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to fine-tuned YOLO11 model (best.pt). If provided, uses unified YOLO11 detector instead of separate detectors."
    )
    
    args = parser.parse_args()
    
    process_document(
        input_path=args.pdf,
        output_dir=args.output,
        threshold_signature=args.threshold_signature,
        threshold_qr=args.threshold_qr,
        threshold_stamp=args.threshold_stamp,
        device=args.device,
        save_json=args.save_json,
        save_images=args.save_images,
        hf_token=args.hf_token,
        model_path=args.model
    )


if __name__ == "__main__":
    main()


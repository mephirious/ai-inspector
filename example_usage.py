#!/usr/bin/env python3
"""Example usage of Digital Inspector."""

import json
from pathlib import Path
from digital_inspector.detectors import SignatureDetector, QRDetector, StampDetector
from digital_inspector.utils.pdf_utils import pdf_to_images
from digital_inspector.utils.image_utils import pil_to_numpy, load_image
from digital_inspector.utils.merge_utils import merge_detections, format_output
from digital_inspector.utils.viz_utils import save_annotated_image


def example_process_image(image_path: str):
    """Example: Process a single image."""
    print(f"Processing image: {image_path}")
    
    # Initialize detectors
    print("Initializing detectors...")
    sig_detector = SignatureDetector(device="auto")
    qr_detector = QRDetector()
    stamp_detector = StampDetector(device="auto")
    
    # Load image
    image_np = load_image(image_path)
    page_height, page_width = image_np.shape[:2]
    print(f"Image size: {page_width}x{page_height}")
    
    # Run all detectors
    all_detections = []
    
    print("\nRunning detection...")
    sig_results = sig_detector.detect(image_np)
    print(f"  Signatures: {len(sig_results)}")
    all_detections.extend(sig_results)
    
    qr_results = qr_detector.detect(image_np)
    print(f"  QR codes: {len(qr_results)}")
    all_detections.extend(qr_results)
    
    stamp_results = stamp_detector.detect(image_np)
    print(f"  Stamps: {len(stamp_results)}")
    all_detections.extend(stamp_results)
    
    # Merge detections
    print("\nMerging detections...")
    merged = merge_detections(all_detections, page_width, page_height)
    print(f"Total unique detections: {len(merged)}")
    
    # Format output
    result = format_output(
        Path(image_path).stem, 1, page_width, page_height, merged
    )
    
    # Print results
    print("\n" + "="*50)
    print("Detection Results:")
    print("="*50)
    print(json.dumps(result, indent=2))
    
    # Save annotated image
    output_path = f"output/{Path(image_path).stem}_annotated.png"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_annotated_image(image_np, output_path, merged, page_width, page_height)
    print(f"\nSaved annotated image: {output_path}")
    
    return result


def example_process_pdf(pdf_path: str):
    """Example: Process a PDF document."""
    print(f"Processing PDF: {pdf_path}")
    
    # Initialize detectors
    print("Initializing detectors...")
    sig_detector = SignatureDetector(device="auto")
    qr_detector = QRDetector()
    stamp_detector = StampDetector(device="auto")
    
    # Convert PDF to images
    pages = pdf_to_images(pdf_path)
    print(f"Found {len(pages)} page(s)")
    
    all_results = []
    
    for page_image, page_num in pages:
        print(f"\nProcessing page {page_num}...")
        image_np = pil_to_numpy(page_image)
        page_width, page_height = page_image.size
        
        # Run all detectors
        all_detections = []
        
        sig_results = sig_detector.detect(image_np)
        all_detections.extend(sig_results)
        
        qr_results = qr_detector.detect(image_np)
        all_detections.extend(qr_results)
        
        stamp_results = stamp_detector.detect(image_np)
        all_detections.extend(stamp_results)
        
        # Merge detections
        merged = merge_detections(all_detections, page_width, page_height)
        print(f"  Found {len(merged)} detection(s)")
        
        # Format output
        result = format_output(
            Path(pdf_path).stem, page_num, page_width, page_height, merged
        )
        all_results.append(result)
        
        # Save annotated image
        output_path = f"output/{Path(pdf_path).stem}/page_{page_num}_annotated.png"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        save_annotated_image(image_np, output_path, merged, page_width, page_height)
    
    # Save JSON results
    json_path = f"output/{Path(pdf_path).stem}/results.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved JSON results: {json_path}")
    
    return all_results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python example_usage.py <image_or_pdf_path>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    if Path(input_path).suffix.lower() == '.pdf':
        example_process_pdf(input_path)
    else:
        example_process_image(input_path)


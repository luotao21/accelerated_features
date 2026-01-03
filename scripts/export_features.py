#!/usr/bin/env python3
"""
Export XFeat features for book pages.

This script extracts XFeat features from book page images and exports them
in a JSON format suitable for mobile app consumption.

Usage:
    python export_features.py --input-dir ./book_pages --output ./book_data
    python export_features.py --isbn 978-0-xxx --input-dir ./pages --output ./output
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch

# Add parent directory to path for module imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.xfeat import XFeat


def round_nested(data, decimals=2):
    """Recursively round all floats in nested structures."""
    if isinstance(data, float):
        return round(data, decimals)
    elif isinstance(data, list):
        return [round_nested(x, decimals) for x in data]
    elif isinstance(data, dict):
        return {k: round_nested(v, decimals) for k, v in data.items()}
    return data


def export_book_features(
    input_dir: str,
    output_dir: str,
    isbn: str = None,
    top_k: int = 1024,  # Reduced from 4096 for smaller file size
    compute_global_descriptor: bool = True,
    precision: int = 2  # Decimal precision for floats
):
    """
    Export XFeat features for all pages in a book.
    
    Args:
        input_dir: Directory containing page images (page-1.jpg, page-2.jpg, etc.)
        output_dir: Output directory for JSON files
        isbn: Book ISBN (optional, will be inferred from directory name if not provided)
        top_k: Number of top keypoints to extract per page (default: 1024)
        compute_global_descriptor: Whether to compute global descriptor for each page
        precision: Decimal precision for floating point numbers (default: 2)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Infer ISBN from directory name if not provided
    if isbn is None:
        isbn = input_path.name
    
    # Initialize XFeat
    print("Initializing XFeat...")
    xfeat = XFeat()
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    image_files = sorted([
        f for f in input_path.iterdir()
        if f.suffix.lower() in image_extensions
    ])
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} page images")
    print(f"Using top_k={top_k}, precision={precision}")
    
    # Process each page
    pages_data = []
    
    for img_file in image_files:
        page_id = img_file.stem  # e.g., "page-1", "cover"
        print(f"Processing {page_id}...")
        
        # Load image
        import cv2
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"  Warning: Could not load {img_file}")
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.tensor(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        # Extract features
        result = xfeat.detectAndCompute(img_tensor, top_k=top_k)[0]
        
        # Convert and round to reduce precision
        keypoints = round_nested(result['keypoints'].cpu().numpy().tolist(), 0)  # Integer coords
        descriptors = round_nested(result['descriptors'].cpu().numpy().tolist(), precision)
        scores = round_nested(result['scores'].cpu().numpy().tolist(), precision)
        
        page_data = {
            "pageId": page_id,
            "imageSize": [img.shape[1], img.shape[0]],  # [width, height]
            "features": {
                "keypoints": keypoints,
                "descriptors": descriptors,
                "scores": scores
            }
        }
        
        # Compute global descriptor (mean of all descriptors)
        if compute_global_descriptor and len(descriptors) > 0:
            global_desc = round_nested(np.mean(result['descriptors'].cpu().numpy(), axis=0).tolist(), precision + 2)
            page_data["globalDescriptor"] = global_desc
        
        pages_data.append(page_data)
        print(f"  Extracted {len(keypoints)} keypoints")
    
    # Create features.json with compact format
    features_output = {
        "isbn": isbn,
        "version": "1.0.0",
        "updatedAt": datetime.now(timezone.utc).isoformat(),
        "pages": pages_data
    }
    
    features_file = output_path / "features.json"
    with open(features_file, 'w') as f:
        json.dump(features_output, f, separators=(',', ':'))  # Compact format
    
    print(f"\nExported features to {features_file}")
    print(f"  Total pages: {len(pages_data)}")
    
    # Print file size
    file_size = features_file.stat().st_size
    if file_size > 1024 * 1024:
        print(f"  File size: {file_size / 1024 / 1024:.2f} MB")
    else:
        print(f"  File size: {file_size / 1024:.2f} KB")


def main():
    parser = argparse.ArgumentParser(description="Export XFeat features for book pages")
    parser.add_argument(
        "--input-dir", "-i",
        required=True,
        help="Directory containing page images"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for JSON files"
    )
    parser.add_argument(
        "--isbn",
        help="Book ISBN (optional, inferred from directory name if not provided)"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=4096,
        help="Number of top keypoints to extract per page (default: 4096)"
    )
    parser.add_argument(
        "--no-global-descriptor",
        action="store_true",
        help="Skip computing global descriptor for each page"
    )
    
    args = parser.parse_args()
    
    export_book_features(
        input_dir=args.input_dir,
        output_dir=args.output,
        isbn=args.isbn,
        top_k=args.top_k,
        compute_global_descriptor=not args.no_global_descriptor
    )


if __name__ == "__main__":
    main()

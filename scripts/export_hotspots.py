#!/usr/bin/env python3
"""
Export hotspots data from the hotspot editor.

This script converts hotspot definitions (polygons + audio mappings) 
to JSON format for mobile app consumption.

Note: In production, the server-side editor will output JSON directly.
This script is for testing and migration purposes.

Usage:
    python export_hotspots.py --input ./hotspots_raw --output ./book_data
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime, timezone


def export_hotspots(
    input_dir: str,
    output_dir: str,
    isbn: str = None,
    audio_base_url: str = None
):
    """
    Export hotspots data to JSON format.
    
    This function processes hotspot definitions and creates a hotspots.json file.
    
    Args:
        input_dir: Directory containing hotspot definition files
        output_dir: Output directory for JSON file
        isbn: Book ISBN
        audio_base_url: Base URL for audio files (e.g., https://cdn.example.com/audio/)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Infer ISBN from directory name if not provided
    if isbn is None:
        isbn = input_path.name
    
    # In production, the editor outputs JSON directly
    # This is a template for the expected format
    
    # Check if there's already a hotspots definition file
    hotspots_file = input_path / "hotspots.json"
    
    if hotspots_file.exists():
        # Load existing hotspots
        with open(hotspots_file, 'r') as f:
            hotspots_data = json.load(f)
        print(f"Loaded existing hotspots from {hotspots_file}")
    else:
        # Create template structure
        hotspots_data = {
            "isbn": isbn,
            "version": "1.0.0",
            "updatedAt": datetime.now(timezone.utc).isoformat(),
            "pages": [
                {
                    "pageId": "page-1",
                    "hotspots": [
                        {
                            "name": "example",
                            "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]],
                            "audioFile": "example.mp3"
                        }
                    ]
                }
            ],
            "audioBaseUrl": audio_base_url or "https://cdn.example.com/books/{isbn}/audio/"
        }
        print("Created template hotspots structure")
    
    # Update metadata
    hotspots_data["isbn"] = isbn
    hotspots_data["updatedAt"] = datetime.now(timezone.utc).isoformat()
    if audio_base_url:
        hotspots_data["audioBaseUrl"] = audio_base_url
    
    # Save to output
    output_file = output_path / "hotspots.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(hotspots_data, f, indent=2, ensure_ascii=False)
    
    print(f"Exported hotspots to {output_file}")
    
    # Count statistics
    total_hotspots = sum(len(p.get("hotspots", [])) for p in hotspots_data.get("pages", []))
    print(f"  Total pages: {len(hotspots_data.get('pages', []))}")
    print(f"  Total hotspots: {total_hotspots}")


def main():
    parser = argparse.ArgumentParser(description="Export hotspots to JSON format")
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Directory containing hotspot definitions"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for JSON file"
    )
    parser.add_argument(
        "--isbn",
        help="Book ISBN"
    )
    parser.add_argument(
        "--audio-base-url",
        help="Base URL for audio files"
    )
    
    args = parser.parse_args()
    
    export_hotspots(
        input_dir=args.input,
        output_dir=args.output,
        isbn=args.isbn,
        audio_base_url=args.audio_base_url
    )


if __name__ == "__main__":
    main()

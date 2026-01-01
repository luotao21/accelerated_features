"""
Hotspot Loader Module

Loads SVG hotspot files and converts them to OpenCV-compatible polygons.
"""

import os
import glob
import numpy as np
import cv2
from svgpathtools import svg2paths


def parse_svg_hotspots(svg_dir: str, target_width: int = None, target_height: int = None):
    """
    Load all SVG files from directory and extract hotspot polygons.
    
    Args:
        svg_dir: Path to directory containing SVG files
        target_width: Target width to scale polygons (None = use original SVG size)
        target_height: Target height to scale polygons (None = use original SVG size)
    
    Returns:
        dict: {name: polygon_points} where polygon_points is np.array of shape (N, 2)
    """
    hotspots = {}
    svg_files = glob.glob(os.path.join(svg_dir, "*.svg"))
    
    for svg_file in svg_files:
        name = os.path.splitext(os.path.basename(svg_file))[0]
        try:
            paths, attributes = svg2paths(svg_file)
            
            if not paths:
                continue
            
            # Get SVG viewBox dimensions from file
            svg_width, svg_height = 2240, 1344  # Default from our SVGs
            
            # Sample path to polygon
            all_points = []
            for path in paths:
                points = sample_path_to_polygon(path, num_samples=100)
                if len(points) > 0:
                    all_points.extend(points)
            
            if all_points:
                points_array = np.array(all_points, dtype=np.float32)
                
                # Scale to target resolution if specified
                if target_width and target_height:
                    points_array[:, 0] *= target_width / svg_width
                    points_array[:, 1] *= target_height / svg_height
                
                hotspots[name] = points_array.astype(np.int32)
                
        except Exception as e:
            print(f"Warning: Failed to parse {svg_file}: {e}")
    
    return hotspots


def sample_path_to_polygon(path, num_samples=100):
    """
    Sample points along an SVG path to create a polygon.
    
    Args:
        path: svgpathtools Path object
        num_samples: Number of points to sample
    
    Returns:
        list of (x, y) tuples
    """
    points = []
    
    try:
        length = path.length()
        if length == 0:
            return points
        
        for i in range(num_samples):
            t = i / num_samples
            point = path.point(t)
            points.append((point.real, point.imag))
            
    except Exception:
        # Fallback: try to get start/end points of segments
        for segment in path:
            try:
                points.append((segment.start.real, segment.start.imag))
            except:
                pass
    
    return points


def create_hotspot_mask(hotspots: dict, width: int, height: int, 
                        color=(0, 255, 0), alpha=128):
    """
    Create a semi-transparent mask image with all hotspots filled.
    
    Args:
        hotspots: dict from parse_svg_hotspots()
        width: Mask width
        height: Mask height
        color: BGR color tuple for hotspots
        alpha: Transparency (0-255)
    
    Returns:
        np.array: BGRA mask image
    """
    # Create BGRA mask
    mask = np.zeros((height, width, 4), dtype=np.uint8)
    
    for name, points in hotspots.items():
        if len(points) < 3:
            continue
        
        # Reshape points for cv2.fillPoly
        pts = points.reshape((-1, 1, 2))
        
        # Fill polygon with semi-transparent color
        cv2.fillPoly(mask, [pts], (*color, alpha))
    
    return mask


def blend_hotspot_overlay(frame: np.ndarray, hotspot_mask: np.ndarray):
    """
    Blend hotspot mask onto a frame.
    
    Args:
        frame: BGR image
        hotspot_mask: BGRA mask from create_hotspot_mask()
    
    Returns:
        np.array: BGR image with hotspots overlaid
    """
    # Resize mask if needed
    if frame.shape[:2] != hotspot_mask.shape[:2]:
        hotspot_mask = cv2.resize(hotspot_mask, (frame.shape[1], frame.shape[0]))
    
    # Extract alpha channel
    alpha = hotspot_mask[:, :, 3:4] / 255.0
    rgb = hotspot_mask[:, :, :3]
    
    # Blend
    result = frame.astype(np.float32) * (1 - alpha) + rgb.astype(np.float32) * alpha
    
    return result.astype(np.uint8)


def warp_hotspot_mask(hotspot_mask: np.ndarray, H: np.ndarray, 
                      output_size: tuple):
    """
    Warp hotspot mask using homography matrix.
    
    Args:
        hotspot_mask: BGRA mask
        H: 3x3 homography matrix
        output_size: (width, height) of output
    
    Returns:
        np.array: Warped BGRA mask
    """
    if H is None:
        return None
    
    return cv2.warpPerspective(hotspot_mask, H, output_size)


def point_in_hotspot(point: tuple, hotspots: dict) -> str:
    """
    Check if a point is inside any hotspot polygon.
    
    Args:
        point: (x, y) coordinates
        hotspots: dict from parse_svg_hotspots()
    
    Returns:
        Name of the hotspot containing the point, or None.
    """
    if point is None:
        return None
    
    x, y = point
    for name, polygon in hotspots.items():
        if len(polygon) < 3:
            continue
        
        # Use cv2.pointPolygonTest
        pts = polygon.reshape((-1, 1, 2))
        result = cv2.pointPolygonTest(pts, (float(x), float(y)), False)
        if result >= 0:  # Inside or on edge
            return name
    
    return None


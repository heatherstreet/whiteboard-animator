#!/usr/bin/env python3
"""
Whiteboard Animation Generator v3

Natural stroke-based reveal - traces contours as if being drawn in realtime.
Follows actual stroke paths for shapes and lines.

Usage:
    python3 whiteboard_animator_v3.py input_image.jpg output.mp4 [options]
"""

import cv2
import numpy as np
from PIL import Image
import subprocess
import os
import sys
import argparse
import tempfile
from collections import deque

def convert_to_sketch(image_path, output_path):
    """Convert an image to a pencil sketch style."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    inverted_blurred = cv2.bitwise_not(blurred)
    sketch = cv2.divide(gray, inverted_blurred, scale=256.0)
    sketch = cv2.normalize(sketch, None, 0, 255, cv2.NORM_MINMAX)
    
    cv2.imwrite(output_path, sketch)
    return sketch

def get_contour_start_point(contour):
    """Get a good starting point for drawing a contour (topmost-leftmost)."""
    points = contour.reshape(-1, 2)
    # Find topmost point, break ties with leftmost
    min_y = np.min(points[:, 1])
    top_points = points[points[:, 1] == min_y]
    start_idx = np.argmin(top_points[:, 0])
    start_point = top_points[start_idx]
    
    # Find index in original contour
    for i, pt in enumerate(points):
        if np.array_equal(pt, start_point):
            return i
    return 0

def order_contours_for_drawing(contours, img_shape):
    """Order contours in natural drawing order (top-to-bottom, left-to-right)."""
    if not contours:
        return []
    
    # Get bounding box center for each contour
    contour_info = []
    for i, cnt in enumerate(contours):
        if len(cnt) < 3:
            continue
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w//2, y + h//2
        contour_info.append((i, cx, cy, cnt))
    
    # Sort by y first (top to bottom), then x (left to right)
    # Use buckets for y to group roughly same-height items
    h = img_shape[0]
    bucket_size = h // 10  # 10 horizontal bands
    
    contour_info.sort(key=lambda x: (x[2] // bucket_size, x[1]))
    
    return [info[3] for info in contour_info]

def trace_contour_pixels(contour, img_shape):
    """Get all pixels along a contour in drawing order."""
    h, w = img_shape[:2]
    
    # Reorder contour to start from a natural point
    points = contour.reshape(-1, 2)
    start_idx = get_contour_start_point(contour)
    points = np.roll(points, -start_idx, axis=0)
    
    # Create a dense path by interpolating between contour points
    path_pixels = []
    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i + 1) % len(points)]
        
        # Get all pixels on line between p1 and p2
        num_steps = max(abs(p2[0] - p1[0]), abs(p2[1] - p1[1]), 1)
        for t in range(num_steps + 1):
            x = int(p1[0] + (p2[0] - p1[0]) * t / num_steps)
            y = int(p1[1] + (p2[1] - p1[1]) * t / num_steps)
            if 0 <= x < w and 0 <= y < h:
                path_pixels.append((x, y))
    
    return path_pixels

def create_stroke_order_map(sketch_img):
    """Create a map showing pixel reveal order based on stroke tracing."""
    h, w = sketch_img.shape[:2]
    if len(sketch_img.shape) == 3:
        gray = cv2.cvtColor(sketch_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = sketch_img.copy()
    
    # Threshold to get content
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Find edges for contour detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate edges slightly to connect nearby strokes
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    # Filter out tiny contours (noise)
    min_contour_len = 20
    contours = [c for c in contours if len(c) >= min_contour_len]
    
    # Order contours for natural drawing
    ordered_contours = order_contours_for_drawing(contours, (h, w))
    
    print(f"  Found {len(ordered_contours)} strokes to trace")
    
    # Create order map
    order_map = np.full((h, w), -1, dtype=np.float32)
    
    current_order = 0
    total_pixels = 0
    
    # Trace each contour
    for contour in ordered_contours:
        path = trace_contour_pixels(contour, (h, w))
        for (x, y) in path:
            if order_map[y, x] < 0:  # Not yet assigned
                order_map[y, x] = current_order
                current_order += 1
                total_pixels += 1
    
    # Also include filled areas (not just edges)
    # Use distance transform from edges to fill interiors
    content_mask = binary > 0
    edge_mask = order_map >= 0
    
    # For pixels that have content but no order yet, assign based on nearest edge
    unfilled = content_mask & ~edge_mask
    if np.any(unfilled):
        # Distance transform from edges
        dist_from_edge = cv2.distanceTransform((~edge_mask).astype(np.uint8) * 255, cv2.DIST_L2, 5)
        
        # Assign unfilled content pixels order based on nearby edge order + distance
        unfilled_coords = np.argwhere(unfilled)
        for (y, x) in unfilled_coords:
            # Find nearest pixel with an order
            search_radius = int(dist_from_edge[y, x]) + 5
            y_min, y_max = max(0, y - search_radius), min(h, y + search_radius)
            x_min, x_max = max(0, x - search_radius), min(w, x + search_radius)
            
            region = order_map[y_min:y_max, x_min:x_max]
            valid = region >= 0
            if np.any(valid):
                # Use mean order of nearby assigned pixels + small offset
                nearby_order = np.mean(region[valid])
                order_map[y, x] = nearby_order + dist_from_edge[y, x] * 0.1
    
    # Normalize to 0-1
    valid_mask = order_map >= 0
    if np.any(valid_mask):
        min_order = np.min(order_map[valid_mask])
        max_order = np.max(order_map[valid_mask])
        if max_order > min_order:
            order_map = np.where(valid_mask, (order_map - min_order) / (max_order - min_order), -1)
    
    return order_map

def generate_frames(sketch_path, output_dir, num_frames, width, height, ease_drawing=True):
    """Generate animation frames with natural stroke-based reveal."""
    
    sketch = cv2.imread(sketch_path)
    if sketch is None:
        raise ValueError(f"Could not load sketch: {sketch_path}")
    
    sketch = cv2.resize(sketch, (width, height), interpolation=cv2.INTER_LANCZOS4)
    white_bg = np.full_like(sketch, 255)
    
    print("Analyzing stroke paths...")
    order_map = create_stroke_order_map(sketch)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for frame_num in range(num_frames):
        progress = frame_num / (num_frames - 1) if num_frames > 1 else 1
        
        if ease_drawing:
            # Ease-out cubic for more natural feel (fast start, slow finish)
            eased = 1 - pow(1 - progress, 3)
            progress = eased
        
        # Reveal mask
        reveal_mask = (order_map >= 0) & (order_map <= progress)
        reveal_mask = reveal_mask.astype(np.uint8) * 255
        
        # Slight feathering for smooth drawing edge
        reveal_mask = cv2.GaussianBlur(reveal_mask, (3, 3), 0)
        
        # Blend
        mask_3ch = cv2.cvtColor(reveal_mask, cv2.COLOR_GRAY2BGR).astype(float) / 255
        frame = (sketch * mask_3ch + white_bg * (1 - mask_3ch)).astype(np.uint8)
        
        frame_path = os.path.join(output_dir, f"frame_{frame_num:06d}.png")
        cv2.imwrite(frame_path, frame)
        
        if frame_num % 60 == 0:
            print(f"  Frame {frame_num}/{num_frames} ({100*frame_num/num_frames:.1f}%)")
    
    print("Frame generation complete!")

def compile_video(frames_dir, output_path, fps):
    """Compile frames into video using ffmpeg."""
    frame_pattern = os.path.join(frames_dir, "frame_%06d.png")
    
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', frame_pattern,
        '-c:v', 'libx264',
        '-profile:v', 'high',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        output_path
    ]
    
    print(f"Compiling video: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"Video saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate whiteboard-style animation v3')
    parser.add_argument('input', help='Input image path')
    parser.add_argument('output', help='Output video path')
    parser.add_argument('--duration', type=float, default=8, help='Video duration in seconds')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    parser.add_argument('--width', type=int, default=1920, help='Output width')
    parser.add_argument('--height', type=int, default=1080, help='Output height')
    parser.add_argument('--no-sketch', action='store_true', help='Keep original style')
    parser.add_argument('--no-ease', action='store_true', help='Disable easing')
    parser.add_argument('--keep-frames', action='store_true', help='Keep temp frames')
    
    args = parser.parse_args()
    
    temp_dir = tempfile.mkdtemp(prefix='whiteboard_')
    frames_dir = os.path.join(temp_dir, 'frames')
    
    try:
        if args.no_sketch:
            sketch_path = args.input
        else:
            sketch_path = os.path.join(temp_dir, 'sketch.png')
            print("Converting to sketch...")
            convert_to_sketch(args.input, sketch_path)
            print(f"Sketch saved to: {sketch_path}")
        
        num_frames = int(args.duration * args.fps)
        
        print(f"Generating {num_frames} frames...")
        generate_frames(
            sketch_path, 
            frames_dir, 
            num_frames, 
            args.width, 
            args.height,
            ease_drawing=not args.no_ease
        )
        
        compile_video(frames_dir, args.output, args.fps)
        
        print(f"\n✓ Animation complete!")
        
        if args.keep_frames:
            print(f"Frames kept in: {frames_dir}")
        
    finally:
        if not args.keep_frames:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == '__main__':
    main()

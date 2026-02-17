#!/usr/bin/env python3
"""
Whiteboard Animation Generator v2

Natural hand-drawn reveal - traces content as if being written in realtime.
No hand overlay, follows actual stroke paths.

Usage:
    python3 whiteboard_animator_v2.py input_image.jpg output.mp4 [options]
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
    
    # Increase contrast
    sketch = cv2.normalize(sketch, None, 0, 255, cv2.NORM_MINMAX)
    
    cv2.imwrite(output_path, sketch)
    return sketch

def find_text_lines(binary_img):
    """Find horizontal text lines by analyzing row density."""
    # Get horizontal projection (sum of black pixels per row)
    projection = np.sum(binary_img < 200, axis=1)
    
    # Find lines by detecting peaks in projection
    threshold = np.max(projection) * 0.05
    in_line = False
    lines = []
    line_start = 0
    
    for i, val in enumerate(projection):
        if val > threshold and not in_line:
            in_line = True
            line_start = i
        elif val <= threshold and in_line:
            in_line = False
            if i - line_start > 5:  # Min line height
                lines.append((line_start, i))
    
    if in_line:
        lines.append((line_start, len(projection) - 1))
    
    return lines

def find_content_in_line(binary_img, y_start, y_end):
    """Find content segments within a line (left to right)."""
    line_region = binary_img[y_start:y_end, :]
    projection = np.sum(line_region < 200, axis=0)
    
    threshold = np.max(projection) * 0.1 if np.max(projection) > 0 else 0
    
    # Find content spans
    segments = []
    in_content = False
    seg_start = 0
    
    for i, val in enumerate(projection):
        if val > threshold and not in_content:
            in_content = True
            seg_start = i
        elif val <= threshold and in_content:
            in_content = False
            segments.append((seg_start, i))
    
    if in_content:
        segments.append((seg_start, len(projection) - 1))
    
    return segments

def create_drawing_order_map(sketch_img):
    """Create a map showing the order in which pixels should appear."""
    h, w = sketch_img.shape[:2]
    if len(sketch_img.shape) == 3:
        gray = cv2.cvtColor(sketch_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = sketch_img.copy()
    
    # Find text lines
    lines = find_text_lines(gray)
    
    if not lines:
        # Fallback: treat entire image as one line
        lines = [(0, h)]
    
    # Create order map (0 = first to draw, higher = later)
    order_map = np.full((h, w), -1, dtype=np.float32)
    
    total_content_width = 0
    line_data = []
    
    for y_start, y_end in lines:
        segments = find_content_in_line(gray, y_start, y_end)
        line_width = sum(seg[1] - seg[0] for seg in segments)
        total_content_width += line_width
        line_data.append((y_start, y_end, segments, line_width))
    
    # Assign order values
    current_order = 0
    
    for y_start, y_end, segments, line_width in line_data:
        for seg_start, seg_end in segments:
            for x in range(seg_start, seg_end):
                for y in range(y_start, y_end):
                    if gray[y, x] < 240:  # Has content
                        order_map[y, x] = current_order
                current_order += 1
    
    # Normalize to 0-1
    max_order = np.max(order_map)
    if max_order > 0:
        order_map = np.where(order_map >= 0, order_map / max_order, -1)
    
    return order_map

def create_stroke_order_map(sketch_img):
    """Create drawing order by tracing actual strokes."""
    h, w = sketch_img.shape[:2]
    if len(sketch_img.shape) == 3:
        gray = cv2.cvtColor(sketch_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = sketch_img.copy()
    
    # Threshold to binary
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    content_mask = binary < 255  # Where there's content
    
    # Find text lines first
    lines = find_text_lines(gray)
    if not lines:
        lines = [(0, h)]
    
    # Create order map
    order_map = np.full((h, w), -1, dtype=np.float32)
    
    current_order = 0
    total_pixels = np.sum(content_mask)
    
    for y_start, y_end in lines:
        # Process this line left to right, with some vertical spread
        for x in range(w):
            col_has_content = False
            for y in range(y_start, y_end):
                if content_mask[y, x] and order_map[y, x] < 0:
                    order_map[y, x] = current_order
                    col_has_content = True
            if col_has_content:
                current_order += 1
    
    # Normalize
    max_order = np.max(order_map)
    if max_order > 0:
        order_map = np.where(order_map >= 0, order_map / max_order, -1)
    
    return order_map

def generate_frames(sketch_path, output_dir, num_frames, width, height, ease_drawing=True):
    """Generate animation frames with natural drawing reveal."""
    
    # Load sketch
    sketch = cv2.imread(sketch_path)
    if sketch is None:
        raise ValueError(f"Could not load sketch: {sketch_path}")
    
    # Resize to target dimensions
    sketch = cv2.resize(sketch, (width, height), interpolation=cv2.INTER_LANCZOS4)
    
    # Create white background
    white_bg = np.full_like(sketch, 255)
    
    # Create drawing order map
    print("Analyzing drawing structure...")
    order_map = create_stroke_order_map(sketch)
    
    # Generate frames
    os.makedirs(output_dir, exist_ok=True)
    
    for frame_num in range(num_frames):
        progress = frame_num / (num_frames - 1) if num_frames > 1 else 1
        
        # Apply easing for more natural feel
        if ease_drawing:
            # Ease-in-out cubic
            if progress < 0.5:
                eased = 4 * progress * progress * progress
            else:
                eased = 1 - pow(-2 * progress + 2, 3) / 2
            progress = eased
        
        # Create mask for what's visible at this progress
        reveal_mask = (order_map >= 0) & (order_map <= progress)
        reveal_mask = reveal_mask.astype(np.uint8) * 255
        
        # Slight feathering at the drawing edge for smoothness
        kernel = np.ones((3, 3), np.uint8)
        reveal_mask = cv2.dilate(reveal_mask, kernel, iterations=1)
        reveal_mask = cv2.GaussianBlur(reveal_mask, (5, 5), 0)
        
        # Blend
        mask_3ch = cv2.cvtColor(reveal_mask, cv2.COLOR_GRAY2BGR).astype(float) / 255
        frame = (sketch * mask_3ch + white_bg * (1 - mask_3ch)).astype(np.uint8)
        
        # Save frame
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
    parser = argparse.ArgumentParser(description='Generate whiteboard-style animation')
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
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix='whiteboard_')
    frames_dir = os.path.join(temp_dir, 'frames')
    
    try:
        # Convert to sketch or use original
        if args.no_sketch:
            sketch_path = args.input
        else:
            sketch_path = os.path.join(temp_dir, 'sketch.png')
            print("Converting to sketch...")
            convert_to_sketch(args.input, sketch_path)
            print(f"Sketch saved to: {sketch_path}")
        
        # Calculate total frames
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
        
        # Compile video
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

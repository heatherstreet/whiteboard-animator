#!/usr/bin/env python3
"""
Whiteboard Animation Generator

Converts a static image into a whiteboard-style animation video with a 
hand drawing effect.

Usage:
    python3 whiteboard_animator.py input_image.jpg output.mp4 [options]

Options:
    --duration SECONDS    Video duration (default: 12)
    --fps FPS             Frames per second (default: 60)
    --width WIDTH         Output width (default: 1920)
    --height HEIGHT       Output height (default: 1080)
    --hand PATH           Path to hand PNG with transparency
    --no-sketch           Don't convert to sketch style (keep original)
"""

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import subprocess
import os
import sys
import argparse
import tempfile
import shutil

def convert_to_sketch(image_path, output_path):
    """Convert an image to a pencil sketch style."""
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Invert the grayscale image
    inverted = cv2.bitwise_not(gray)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    
    # Invert the blurred image
    inverted_blurred = cv2.bitwise_not(blurred)
    
    # Create the pencil sketch
    sketch = cv2.divide(gray, inverted_blurred, scale=256.0)
    
    # Enhance contrast
    sketch = cv2.normalize(sketch, None, 0, 255, cv2.NORM_MINMAX)
    
    # Convert to 3-channel for consistency
    sketch_bgr = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    
    # Save
    cv2.imwrite(output_path, sketch_bgr)
    return output_path


def extract_hand_from_video(video_path, output_path, frame_num=300):
    """Extract a hand image from a whiteboard video and create transparent PNG."""
    # Extract frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError("Could not extract frame from video")
    
    # The hand is typically in the lower-right area
    # We'll create a mask based on the white background
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create mask for non-white areas (the hand)
    # White has high value and low saturation
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Invert to get hand mask
    hand_mask = cv2.bitwise_not(white_mask)
    
    # Clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    hand_mask = cv2.morphologyEx(hand_mask, cv2.MORPH_CLOSE, kernel)
    hand_mask = cv2.morphologyEx(hand_mask, cv2.MORPH_OPEN, kernel)
    
    # Find the hand region (bottom-right quadrant typically)
    h, w = frame.shape[:2]
    
    # Focus on bottom-right for hand
    roi_mask = np.zeros_like(hand_mask)
    roi_mask[h//3:, w//3:] = 255
    hand_mask = cv2.bitwise_and(hand_mask, roi_mask)
    
    # Find contours and get the largest one (the hand)
    contours, _ = cv2.findContours(hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(largest)
        
        # Expand bounds slightly
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        cw = min(w - x, cw + 2 * padding)
        ch = min(h - y, ch + 2 * padding)
        
        # Crop the hand region
        hand_crop = frame[y:y+ch, x:x+cw]
        mask_crop = hand_mask[y:y+ch, x:x+cw]
        
        # Create RGBA image with transparency
        rgba = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = mask_crop
        
        cv2.imwrite(output_path, rgba)
        return output_path, (x, y, cw, ch)
    
    raise ValueError("Could not find hand in video frame")


def create_default_hand(output_path, size=(150, 200)):
    """Create a simple placeholder hand+pen image."""
    # Create a simple stylized hand shape
    img = Image.new('RGBA', size, (0, 0, 0, 0))
    
    # We'll use numpy to draw a simple shape
    arr = np.array(img)
    
    # Draw a simple pen shape
    cv_img = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    
    # Pen body (pink/peach color)
    cv2.rectangle(cv_img, (60, 20), (90, 150), (180, 150, 200, 255), -1)
    
    # Pen tip
    pts = np.array([[70, 150], [80, 180], [90, 150]], np.int32)
    cv2.fillPoly(cv_img, [pts], (100, 100, 100, 255))
    
    # Simple hand shape (skin tone)
    cv2.ellipse(cv_img, (75, 80), (40, 60), 0, 0, 360, (200, 180, 170, 255), -1)
    
    cv2.imwrite(output_path, cv_img)
    return output_path


def generate_frames(sketch_path, hand_path, output_dir, duration=12, fps=60, 
                    width=1920, height=1080, pen_tip_offset=(0, 0)):
    """Generate all frames for the animation."""
    
    total_frames = int(duration * fps)
    
    # Load sketch
    sketch = cv2.imread(sketch_path)
    if sketch is None:
        raise ValueError(f"Could not load sketch: {sketch_path}")
    
    # Resize sketch to fit output dimensions while maintaining aspect ratio
    sketch_h, sketch_w = sketch.shape[:2]
    scale = min(width / sketch_w, height / sketch_h) * 0.85  # 85% of frame
    new_w = int(sketch_w * scale)
    new_h = int(sketch_h * scale)
    sketch = cv2.resize(sketch, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Center the sketch
    offset_x = (width - new_w) // 2
    offset_y = (height - new_h) // 2
    
    # Load hand with alpha
    hand = cv2.imread(hand_path, cv2.IMREAD_UNCHANGED)
    if hand is None:
        print(f"Warning: Could not load hand image, creating placeholder")
        create_default_hand(hand_path)
        hand = cv2.imread(hand_path, cv2.IMREAD_UNCHANGED)
    
    # Resize hand to reasonable size
    hand_scale = 0.4
    hand = cv2.resize(hand, None, fx=hand_scale, fy=hand_scale, interpolation=cv2.INTER_LANCZOS4)
    hand_h, hand_w = hand.shape[:2]
    
    # Pen tip position relative to hand image (adjust based on actual hand)
    pen_tip_x = hand_w // 2 + pen_tip_offset[0]
    pen_tip_y = hand_h - 20 + pen_tip_offset[1]
    
    print(f"Generating {total_frames} frames...")
    
    for frame_idx in range(total_frames):
        # Progress through the reveal (0.0 to 1.0)
        progress = frame_idx / (total_frames - 1) if total_frames > 1 else 1.0
        
        # Create white background
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Calculate reveal width (left to right)
        reveal_x = int(progress * new_w)
        
        # Copy revealed portion of sketch onto frame
        if reveal_x > 0:
            frame[offset_y:offset_y+new_h, offset_x:offset_x+reveal_x] = sketch[:, :reveal_x]
        
        # Calculate hand position (at the edge of reveal)
        hand_x = offset_x + reveal_x - pen_tip_x
        hand_y = offset_y + (new_h // 2) - pen_tip_y
        
        # Add some natural movement (slight vertical wobble)
        wobble = int(np.sin(frame_idx * 0.3) * 3)
        hand_y += wobble
        
        # Overlay hand onto frame
        if hand.shape[2] == 4:  # Has alpha channel
            # Calculate valid overlay region
            y1 = max(0, hand_y)
            y2 = min(height, hand_y + hand_h)
            x1 = max(0, hand_x)
            x2 = min(width, hand_x + hand_w)
            
            # Corresponding region in hand image
            hy1 = max(0, -hand_y)
            hy2 = hy1 + (y2 - y1)
            hx1 = max(0, -hand_x)
            hx2 = hx1 + (x2 - x1)
            
            if y2 > y1 and x2 > x1:
                hand_roi = hand[hy1:hy2, hx1:hx2]
                frame_roi = frame[y1:y2, x1:x2]
                
                # Alpha blending
                alpha = hand_roi[:, :, 3:4] / 255.0
                hand_rgb = hand_roi[:, :, :3]
                
                blended = (alpha * hand_rgb + (1 - alpha) * frame_roi).astype(np.uint8)
                frame[y1:y2, x1:x2] = blended
        
        # Save frame
        frame_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.png")
        cv2.imwrite(frame_path, frame)
        
        if frame_idx % 60 == 0:
            print(f"  Frame {frame_idx}/{total_frames} ({progress*100:.1f}%)")
    
    print("Frame generation complete!")
    return total_frames


def compile_video(frames_dir, output_path, fps=60):
    """Compile frames into video using ffmpeg."""
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', os.path.join(frames_dir, 'frame_%06d.png'),
        '-c:v', 'libx264',
        '-profile:v', 'high',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        output_path
    ]
    
    print(f"Compiling video: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ffmpeg error: {result.stderr}")
        raise RuntimeError("Video compilation failed")
    
    print(f"Video saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate whiteboard animation from image')
    parser.add_argument('input', help='Input image path')
    parser.add_argument('output', help='Output video path')
    parser.add_argument('--duration', type=float, default=12, help='Video duration in seconds')
    parser.add_argument('--fps', type=int, default=60, help='Frames per second')
    parser.add_argument('--width', type=int, default=1920, help='Output width')
    parser.add_argument('--height', type=int, default=1080, help='Output height')
    parser.add_argument('--hand', help='Path to hand PNG with transparency')
    parser.add_argument('--no-sketch', action='store_true', help='Keep original image style')
    parser.add_argument('--keep-frames', action='store_true', help='Keep intermediate frames')
    
    args = parser.parse_args()
    
    # Create temp directory for frames
    temp_dir = tempfile.mkdtemp(prefix='whiteboard_')
    frames_dir = os.path.join(temp_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)
    
    try:
        # Step 1: Convert to sketch (if needed)
        if args.no_sketch:
            sketch_path = args.input
            print(f"Using original image: {sketch_path}")
        else:
            sketch_path = os.path.join(temp_dir, 'sketch.png')
            print(f"Converting to sketch...")
            convert_to_sketch(args.input, sketch_path)
            print(f"Sketch saved to: {sketch_path}")
        
        # Step 2: Setup hand image
        if args.hand and os.path.exists(args.hand):
            hand_path = args.hand
        else:
            hand_path = os.path.join(temp_dir, 'hand.png')
            # Look for default hand in script directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            default_hand = os.path.join(script_dir, 'hand.png')
            if os.path.exists(default_hand):
                shutil.copy(default_hand, hand_path)
            else:
                print("Creating placeholder hand...")
                create_default_hand(hand_path)
        
        print(f"Using hand image: {hand_path}")
        
        # Step 3: Generate frames
        generate_frames(
            sketch_path, hand_path, frames_dir,
            duration=args.duration, fps=args.fps,
            width=args.width, height=args.height
        )
        
        # Step 4: Compile video
        compile_video(frames_dir, args.output, fps=args.fps)
        
        print("\n✓ Animation complete!")
        
    finally:
        if not args.keep_frames:
            shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            print(f"Frames kept in: {frames_dir}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Extract hand with pen from whiteboard video and save as transparent PNG."""

import cv2
import numpy as np
import sys

def extract_hand(video_path, output_path, frame_num=300):
    """Extract hand from whiteboard video frame."""
    
    # Read frame from video
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Error: Could not read frame {frame_num}")
        return False
    
    h, w = frame.shape[:2]
    print(f"Frame size: {w}x{h}")
    
    # Convert to HSV for better segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create mask for white background (high V, low S)
    # White/near-white pixels
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Also detect the sketch lines (dark pixels)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, dark_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    
    # Combine: background is white OR dark sketch lines
    background_mask = cv2.bitwise_or(white_mask, dark_mask)
    
    # Invert to get foreground (hand)
    hand_mask = cv2.bitwise_not(background_mask)
    
    # Clean up mask
    kernel = np.ones((5, 5), np.uint8)
    hand_mask = cv2.morphologyEx(hand_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    hand_mask = cv2.morphologyEx(hand_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Focus on lower-right area where hand typically is
    roi_mask = np.zeros_like(hand_mask)
    roi_mask[h//3:, w//3:] = 255
    hand_mask = cv2.bitwise_and(hand_mask, roi_mask)
    
    # Find largest contour (the hand)
    contours, _ = cv2.findContours(hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Error: No hand found in frame")
        return False
    
    # Get largest contour
    largest = max(contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(largest)
    
    print(f"Hand found at ({x}, {y}) size {cw}x{ch}")
    
    # Add padding
    pad = 30
    x = max(0, x - pad)
    y = max(0, y - pad)
    cw = min(w - x, cw + 2*pad)
    ch = min(h - y, ch + 2*pad)
    
    # Create refined mask for just the bounding region
    hand_region = frame[y:y+ch, x:x+cw]
    mask_region = hand_mask[y:y+ch, x:x+cw]
    
    # Smooth the mask edges
    mask_region = cv2.GaussianBlur(mask_region, (5, 5), 0)
    
    # Create RGBA image
    rgba = cv2.cvtColor(hand_region, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask_region
    
    # Save
    cv2.imwrite(output_path, rgba)
    print(f"Hand saved to: {output_path}")
    print(f"Hand dimensions: {cw}x{ch}")
    
    return True

if __name__ == '__main__':
    video = sys.argv[1] if len(sys.argv) > 1 else "/root/.clawdbot/media/inbound/e770a812-7c87-4dae-a5ea-4d0c22d2e961.mp4"
    output = sys.argv[2] if len(sys.argv) > 2 else "/root/clawd/whiteboard-animator/hand.png"
    frame = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    
    extract_hand(video, output, frame)

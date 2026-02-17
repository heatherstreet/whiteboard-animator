#!/usr/bin/env python3
"""Extract hand with pen from whiteboard video - improved version."""

import cv2
import numpy as np

def extract_hand_clean(video_path, output_path, frame_num=400):
    """Extract hand using skin color detection."""
    
    # Read frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error reading frame")
        return False
    
    h, w = frame.shape[:2]
    
    # Focus on bottom-right quadrant where hand is
    # Based on the video, hand enters from bottom-right
    roi_x, roi_y = int(w * 0.5), int(h * 0.4)
    roi = frame[roi_y:, roi_x:]
    
    # Convert to HSV for skin detection
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Skin color range (covers various skin tones)
    lower_skin = np.array([0, 20, 70])
    upper_skin = np.array([25, 255, 255])
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Also detect the pink pen
    lower_pink = np.array([140, 30, 100])
    upper_pink = np.array([180, 255, 255])
    pink_mask = cv2.inRange(hsv, lower_pink, upper_pink)
    
    # Detect gray sweater
    lower_gray = np.array([0, 0, 50])
    upper_gray = np.array([180, 50, 150])
    gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)
    
    # Combine masks
    combined_mask = cv2.bitwise_or(skin_mask, pink_mask)
    combined_mask = cv2.bitwise_or(combined_mask, gray_mask)
    
    # Clean up
    kernel = np.ones((7, 7), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find largest connected component
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("No hand found")
        return False
    
    # Get the largest contour
    largest = max(contours, key=cv2.contourArea)
    
    # Create a clean mask from just this contour
    clean_mask = np.zeros_like(combined_mask)
    cv2.drawContours(clean_mask, [largest], -1, 255, -1)
    
    # Feather the edges
    clean_mask = cv2.GaussianBlur(clean_mask, (9, 9), 0)
    
    # Get bounding rect
    x, y, cw, ch = cv2.boundingRect(largest)
    
    # Add padding
    pad = 20
    x = max(0, x - pad)
    y = max(0, y - pad)
    cw = min(roi.shape[1] - x, cw + 2*pad)
    ch = min(roi.shape[0] - y, ch + 2*pad)
    
    # Crop
    hand_crop = roi[y:y+ch, x:x+cw]
    mask_crop = clean_mask[y:y+ch, x:x+cw]
    
    # Create RGBA
    rgba = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask_crop
    
    cv2.imwrite(output_path, rgba)
    print(f"Hand extracted: {cw}x{ch}")
    print(f"Saved to: {output_path}")
    
    return True

if __name__ == '__main__':
    video = "/root/.clawdbot/media/inbound/e770a812-7c87-4dae-a5ea-4d0c22d2e961.mp4"
    output = "/root/clawd/whiteboard-animator/hand.png"
    
    # Try different frames to find best one
    for frame_num in [400, 500, 600]:
        print(f"\nTrying frame {frame_num}...")
        extract_hand_clean(video, output, frame_num)

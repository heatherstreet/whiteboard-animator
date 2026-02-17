# Whiteboard Animator

Converts static images into whiteboard-style animation videos with a hand-drawing effect.

## Usage

```bash
python3 whiteboard_animator.py <input_image> <output_video> [options]
```

### Options

- `--duration SECONDS` - Video duration (default: 12)
- `--fps FPS` - Frames per second (default: 60)
- `--width WIDTH` - Output width (default: 1920)
- `--height HEIGHT` - Output height (default: 1080)
- `--hand PATH` - Custom hand PNG with transparency
- `--no-sketch` - Keep original image style (don't convert to sketch)
- `--keep-frames` - Keep intermediate frames for debugging

### Examples

```bash
# Basic usage - convert photo to whiteboard animation
python3 whiteboard_animator.py photo.jpg output.mp4

# Shorter video with custom hand
python3 whiteboard_animator.py logo.png logo_anim.mp4 --duration 8 --hand custom_hand.png

# Keep original colors (no sketch conversion)
python3 whiteboard_animator.py diagram.png diagram_anim.mp4 --no-sketch
```

## Files

- `whiteboard_animator.py` - Main animation script
- `extract_hand.py` - Extract hand from existing whiteboard video
- `hand.png` - Default hand asset (transparent PNG)

## Dependencies

- Python 3
- OpenCV (`python3-opencv`)
- Pillow (`python3-pil`)
- ffmpeg

## How It Works

1. **Sketch Conversion** - Input image is converted to pencil sketch style using edge detection and dodge blending
2. **Progressive Reveal** - Sketch is revealed left-to-right over the video duration
3. **Hand Animation** - Hand overlay follows the reveal edge with slight natural movement
4. **Video Compilation** - Frames are compiled to MP4 using ffmpeg

## TODO

- [ ] Path-based reveal (follow actual lines instead of left-to-right)
- [ ] Multiple hand poses for variety
- [ ] Sound effects option
- [ ] Vertical reveal mode

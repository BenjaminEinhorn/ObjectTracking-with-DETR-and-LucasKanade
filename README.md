# Skier Tracking using DETR and Optical Flow

This project tracks a skier performing a backflip in a video using:
1. **DETR (DEtection TRansformer)**: For detecting the skier in the initial frame.
2. **Pyramidal Lucas-Kanade Optical Flow**: For tracking feature points across subsequent frames.

### Features
- Detects and tracks a skier through dynamic motion.
- Draws bounding boxes, motion paths, and tracks points in real-time.
- Outputs a video with tracked motion visualization.

### How to Use
1. Install dependencies (see `requirements.txt`).
2. Update the `video_path` variable with the path to your input video.
3. Run the script to process the video.
4. Output video will be saved as `skier_tracking.mp4`.

### Dependencies
See `requirements.txt` for detailed package versions.

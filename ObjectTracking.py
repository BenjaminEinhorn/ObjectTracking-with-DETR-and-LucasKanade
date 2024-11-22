"""
Skier Tracking using DETR and Optical Flow
==========================================

This script tracks a skier through a dramatic backflip in a video using:
1. DETR (DEtection TRansformer) for initial object detection.
2. Pyramidal Lucas-Kanade Optical Flow for tracking feature points across frames.

Dependencies:
transformers==4.39.3
torch==2.2.2
opencv-python==4.9.0.80
Pillow==10.3.0
numpy==1.26.4

"""

from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import cv2
import numpy as np

# Load the DETR model and processor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Video file path (update this path as needed)
video_path = r'E:/Computer Vision Class/Capstone/Backflip2.mp4'
cap = cv2.VideoCapture(video_path)
ret, initial_frame = cap.read()

if not ret:
    print("Error: Cannot read video file.")
    exit()

def get_detr_bounding_boxes(frame, threshold=0.7):
    """
    Detect objects in a frame using Hugging Face's DETR model.

    Args:
        frame (numpy.ndarray): Input image (BGR format).
        threshold (float): Confidence threshold for filtering bounding boxes.

    Returns:
        list: List of bounding boxes [(x_min, y_min, x_max, y_max)] for `person` class.
    """
    # Convert the frame (BGR to RGB) and transform it for DETR
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=pil_image, return_tensors="pt").to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(pixel_values=inputs["pixel_values"])

    # Extract predictions
    logits = outputs.logits[0]  # Shape: [num_queries, num_classes]
    boxes = outputs.pred_boxes[0]  # Shape: [num_queries, 4]

    # Convert logits to probabilities
    probs = logits.softmax(-1)[:, :-1]  # Exclude the "no object" class
    scores, labels = probs.max(dim=1)

    # Filter for `person` class (class_id=1 in COCO) and high confidence
    person_indices = (labels == 1) & (scores > threshold)
    selected_boxes = boxes[person_indices].cpu().numpy()

    # Convert DETR's bounding box format (cx, cy, width, height) to (x_min, y_min, x_max, y_max)
    h, w = frame.shape[:2]
    final_boxes = []
    for box in selected_boxes:
        cx, cy, bw, bh = box
        x_min = int((cx - bw / 2) * w)
        y_min = int((cy - bh / 2) * h)
        x_max = int((cx + bw / 2) * w)
        y_max = int((cy + bh / 2) * h)
        final_boxes.append([x_min, y_min, x_max, y_max])

    return final_boxes

# Detect skier (person) in the initial frame
bounding_boxes = get_detr_bounding_boxes(initial_frame)

if bounding_boxes:
    # Select the first detected bounding box
    skier_box = bounding_boxes[0]
    x_min, y_min, x_max, y_max = skier_box
    print(f"Detected skier's bounding box: {skier_box}")

    # Extract ROI for Shi-Tomasi corner detection
    roi_frame = initial_frame[y_min:y_max, x_min:x_max]
    gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

    # Detect feature points inside the bounding box
    feature_params = dict(maxCorners=100, qualityLevel=0.2, minDistance=7, blockSize=7)
    p0 = cv2.goodFeaturesToTrack(gray_roi, mask=None, **feature_params)

    # Adjust detected points to global coordinates
    if p0 is not None:
        p0[:, 0, 0] += x_min
        p0[:, 0, 1] += y_min
else:
    print("No skier detected in the initial frame.")
    exit()

def pyramidal_lucas_kanade(p0, first_frame, cap):
    """
    Tracks feature points using Pyramidal Lucas-Kanade Optical Flow.

    Args:
        p0 (numpy.ndarray): Initial feature points for tracking.
        first_frame (numpy.ndarray): First video frame.
        cap (cv2.VideoCapture): Video capture object.
    """
    # Optical flow parameters
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(r'E:\Computer Vision Class\Capstone\skier_tracking.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    # Convert first frame to grayscale
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, next_frame = cap.read()
        if not ret:
            break

        # Convert current frame to grayscale
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(first_gray, next_gray, p0, None, **lk_params)

        if p1 is not None:
            # Filter valid points
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            if len(good_new) == 0:
                print("No more points to track. Exiting.")
                break

            # Draw motion paths and yellow points
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = map(int, new.ravel())
                c, d = map(int, old.ravel())
                cv2.line(next_frame, (a, b), (c, d), (0, 255, 0), 2)
                cv2.circle(next_frame, (a, b), 5, (0, 255, 255), -1)

            # Draw bounding box around tracked points
            x_min, y_min = np.min(good_new, axis=0).ravel()
            x_max, y_max = np.max(good_new, axis=0).ravel()
            cv2.rectangle(next_frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)

            # Write the frame to the output video
            out.write(next_frame)

            # Show tracking visualization
            cv2.imshow('Pyramidal Lucas-Kanade Optical Flow', next_frame)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

            # Update previous frame and points
            first_gray = next_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
        else:
            print("No points to track.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Track the skier
pyramidal_lucas_kanade(p0, initial_frame, cap)

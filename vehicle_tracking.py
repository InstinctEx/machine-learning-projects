"""
Dataset: https://www.kaggle.com/datasets/sruizdecastillam/base-de-datos-de-videos-de-transito-vehicular
Model: YOLO11x από Ultralytics με tracking
"""

import cv2
import time
import numpy as np
import torch
from ultralytics import YOLO

# === PARAMETERS ===
VIDEO_PATH    = 'trafficpatisiwn.mp4'        # Input video
OUTPUT_VIDEO  = 'traffic_tracked_roi.mp4'    # Output annotated video
MODEL_WEIGHTS = 'yolo11x.pt'                 # YOLO11x model weights
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'
CONF_THRESH   = 0.25
IOU_THRESH    = 0.45
CLASSES       = ['car','bus','truck','motorbike']
TRACKER       = 'bytetrack.yaml'             # Tracker config file

# === ROI ===
# Either load ROI manually OR uncomment extract_frame_and_define_roi()
ROI_POLYGON = [(726, 2),(45, 1076), (1824, 1076), (1141, 2)]
ROI_CONTOUR = np.array(ROI_POLYGON, dtype=np.int32).reshape((-1, 1, 2))

# === DEVICE CHECK ===
print(f"Χρησιμοποιείται {'GPU: ' + torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'}")

# === LOAD YOLO MODEL ===
model = YOLO(MODEL_WEIGHTS)

# === OPTIONAL: Extract frame and save as image for manual ROI selection ===
def extract_frame_and_define_roi(video_path='trafficpatisiwn.mp4', output_img='frame0.png'):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Δεν βρέθηκε frame")
    cv2.imwrite(output_img, frame)
    print(f"Saved first frame as {output_img} to select ROI manually.")

# === MAIN PROCESSING FUNCTION ===
def process_video():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise IOError(f"Can't open video: {VIDEO_PATH}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30

    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    unique_ids = set()
    start_time = time.time()

    results = model.track(
        source=VIDEO_PATH,
        show=False,
        stream=True,
        tracker=TRACKER,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        device=DEVICE
    )

    for r in results:
        frame = r.orig_img
        boxes     = r.boxes.xyxy.cpu().numpy().astype(int)
        class_ids = r.boxes.cls.cpu().numpy().astype(int)
        track_ids = r.boxes.id.cpu().numpy().astype(int) if hasattr(r.boxes, 'id') and r.boxes.id is not None else [-1] * len(boxes)

        # Draw ROI polygon
        cv2.polylines(frame, [ROI_CONTOUR], isClosed=True, color=(255, 0, 0), thickness=2)

        for bbox, cid, tid in zip(boxes, class_ids, track_ids):
            class_name = model.names[int(cid)]
            if class_name in CLASSES:
                x1, y1, x2, y2 = bbox
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if cv2.pointPolygonTest(ROI_CONTOUR, (float(cx), float(cy)), False) >= 0:
                    if tid >= 0:
                        unique_ids.add(tid)
                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name}-{tid}" if tid >= 0 else class_name
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Overlay count
        cv2.putText(frame, f"Unique Vehicles: {len(unique_ids)}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        out.write(frame)

    cap.release()
    out.release()
    elapsed = time.time() - start_time
    print(f"✅ Unique vehicles in ROI: {len(unique_ids)}")
    print(f"🕒 Elapsed time           : {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")

# === MAIN ENTRY ===
if __name__ == '__main__':
    # Uncomment this to extract a frame for selecting ROI manually
    # extract_frame_and_define_roi()

    process_video()

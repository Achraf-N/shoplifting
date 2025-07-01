import os
import cv2
import pandas as pd
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

# Define the path to your custom model
model_path = r'C:\Users\ASUS\Desktop\Start\crowdhuman_yolov5m.pt'

# Load the model with weights_only=False (only if you trust the source)
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, _verbose=False)
#model = torch.load(model_path, map_location='cpu', weights_only=False)  # Alternative loading

# Set the confidence threshold
model.conf = 0.5
model.classes = [0]  # Only person class

# Init tracker
tracker = DeepSort(max_age=2, n_init=3)

RESULTS = []

def parse_filename(filename):
    parts = filename.replace(".png", "").split("_")
    clip_id = int(parts[0].replace("Shoplifting", "").replace("Normal", ""))
    frame_number = int(parts[-1])
    return clip_id, frame_number

def process_clip_folder(clip_folder_path):
    image_files = sorted(
        [f for f in os.listdir(clip_folder_path) if f.endswith(".png")],
        key=lambda x: int(x.split("_")[-1].replace(".png", ""))
    )

    clip_frames = image_files[:64]

    for file in clip_frames:
        clip_id, frame_number = parse_filename(file)
        full_path = os.path.join(clip_folder_path, file)
        frame = cv2.imread(full_path)

        # Inference
        results = model(frame, size=640)
        detections = []

        for *xyxy, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = map(float, xyxy)
            conf = float(conf)

            # Scale to original size if needed (assuming 256x256 original)
            scale_x = 256 / 640
            scale_y = 256 / 640
            x1 *= scale_x
            y1 *= scale_y
            x2 *= scale_x
            y2 *= scale_y

            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 0))  # class 0 = person

        # DeepSORT tracking
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, r, b = track.to_ltrb()
            w = r - l
            h = b - t
            RESULTS.append([
                clip_id, frame_number, track_id,
                int(l), int(t), int(w), int(h)
            ])

clip_folder = "test1"
process_clip_folder(clip_folder)

# Save to CSV
df = pd.DataFrame(RESULTS, columns=["Clip", "Frame", "Person", "Left", "Top", "Width", "Height"])
df.to_csv("dataTable.csv", index=False)
print("Saved clip tracking results.")
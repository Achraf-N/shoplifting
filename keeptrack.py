#FOR TESTING
import os
import cv2
import pandas as pd
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

model = YOLO("yolov8s.pt")  

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

        frame_resized = cv2.resize(frame, (640, 640))
        results = model(frame_resized, classes=[0])[0]

        detections = []
        for box in results.boxes:
            conf = float(box.conf)
            if conf > 0.5:
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Scale back to 256x256 original image coordinates
                scale_x = 256 / 640
                scale_y = 256 / 640
                x1 *= scale_x
                y1 *= scale_y
                x2 *= scale_x
                y2 *= scale_y

                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 0))  
                
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

            # Optional: Draw for debugging
            # cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
            # cv2.putText(frame, f'ID: {track_id}', (int(l), int(t)-10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Optional: Show debug window
        # cv2.imshow("Tracking", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break



clip_folder = "test1" #it Should be 64 frames inside folder 
process_clip_folder(clip_folder)

# Save results
df = pd.DataFrame(RESULTS, columns=["Clip", "Frame", "Person", "Left", "Top", "Width", "Height"])
df.to_csv("dataTable.csv", index=False)
print("Saved clip tracking results.")

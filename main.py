import os
import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import joblib  # For loading the scaler

# Load trained components
yolo = YOLO("yolov8n.pt")
tracker = SORT(max_lost=3, tracker_output_format='mot_challenge')
lstm_model = load_model("shoplifting_lstm.h5")
scaler = joblib.load("scaler.pkl")  # same one used during training

# Buffers
sequence_buffer = defaultdict(lambda: deque(maxlen=64))
prediction_history = defaultdict(lambda: deque(maxlen=5))  # For smoothing

"""
def extract_features(track, prev_features=None):
    _, track_id, x, y, w, h, *_ = track

    aspect_ratio = w / (h + 1e-6)
    area = w * h
    velocity_x, velocity_y = 0, 0

    if prev_features:
        velocity_x = x - prev_features[0]
        velocity_y = y - prev_features[1]

    return [x, y, w, h, aspect_ratio, area, velocity_x, velocity_y, track_id]

"""

def extract_features(track, prev_features=None):
    _, track_id, x, y, w, h, *_ = track

    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h

    aspect_ratio = w / (h + 1e-6)
    avg_size = (w + h) / 2
    area = w * h

    features = [
        w, h,
        aspect_ratio,
        x1, y1,
        x2, y2,
        x2 - x1, y2 - y1,
        avg_size,
        area
    ]

    return features

def main(video_source=0):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect people
        results = yolo(frame, classes=[0], verbose=False)[0]
        #-----------------------------------------------------------------------
        bboxes, scores, class_ids = [], [], []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bboxes.append([x1, y1, x2 - x1, y2 - y1])  # [x, y, w, h]
            scores.append(float(box.conf[0]))
            class_ids.append(int(box.cls[0]))

        bboxes = np.array(bboxes, dtype=np.float32) if bboxes else np.empty((0, 4))
        scores = np.array(scores, dtype=np.float32) if scores else np.empty((0,))
        class_ids = np.array(class_ids, dtype=np.int32) if class_ids else np.empty((0,))

        # SORT Tracking
        tracks = tracker.update(bboxes, scores, class_ids)

        for track in tracks:
            _, track_id, x, y, w, h, *_ = track
            track_id = int(track_id)

            # Extract features
            prev_feat = sequence_buffer[track_id][-1] if len(sequence_buffer[track_id]) > 0 else None
            features = extract_features(track, prev_feat)

            sequence_buffer[track_id].append(features)

            # Prediction when buffer is full
            if len(sequence_buffer[track_id]) == 64:
                seq = np.array(sequence_buffer[track_id])
                input_seq = seq[:, :11]  # Only features, exclude ID
                scaled_seq = scaler.transform(input_seq).reshape(1, 64, 11)

                pred = lstm_model.predict(scaled_seq, verbose=0)[0]
                prob = pred[0]  # Index 1 → "Shoplifting" class
                prediction_history[track_id].append(prob)

                avg_prob = np.mean(prediction_history[track_id])
                if avg_prob > 0.7:
                    status = "SHOPLIFTING"
                    color = (0, 0, 255)
                elif avg_prob > 0.4:
                    status = "Suspicious"
                    color = (0, 165, 255)
                else:
                    status = "Normal"
                    color = (0, 255, 0)

                # Draw status
                cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
                cv2.putText(frame, f"ID:{track_id} {status}", (int(x), int(y) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                if avg_prob > 0.7:
                    cv2.putText(frame, "SHOPLIFTING ALERT!", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.imshow("Shoplifting Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

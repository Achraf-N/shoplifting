from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

def main(camera_source=0):
    # Initialize models
    model = YOLO("yolov8n.pt") 
    tracker = DeepSort(max_age=30)  
    
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {camera_source}")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        results = model(frame, classes=[0, 24, 26, 28, 43], verbose=False)
        
        knif_detected = any(int(box.cls.item()) == 43 for box in results[0].boxes)
        
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf.item())  
            cls_id = int(box.cls.item())   

            if cls_id == 0:  
                detections.append(([x1, y1, x2-x1, y2-y1], conf, cls_id))
        
        tracked_people = tracker.update_tracks(detections, frame=frame)
        
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls.item())
            label = model.names[cls_id]
            conf = float(box.conf.item())  
            
            if cls_id == 0:  
                color = (0, 0, 255) if knif_detected else (0, 255, 0) 
            else: 
                color = (255, 0, 0)  
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        for track in tracked_people:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            text_color = (0, 0, 255) if knif_detected else (0, 255, 255)  # Red if phone detected, else yellow
            cv2.putText(frame, f"Person {track_id}", (x1, y1-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        if knif_detected:
            cv2.putText(frame, "WARNING: Knives detected!", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("Smart Surveillance", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()  
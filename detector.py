import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict
import time

class TrafficMonitor:
    """AI Traffic Monitoring System - Detection Engine"""
    
    def __init__(self):
        print("🚀 Initializing Traffic Monitor...")
        
        # Load YOLOv8 Nano (fastest for Mac)
        self.model = YOLO('yolov8n.pt')
        print("✅ YOLO model loaded")
        
        # ByteTrack - tracks vehicles across frames
        self.tracker = sv.ByteTrack()
        
        # Drawing tools
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5)
        
        # Counting line (set based on video size)
        self.line_zone = None
        self.line_annotator = sv.LineZoneAnnotator(thickness=3, text_thickness=2)
        
        # Statistics
        self.vehicle_counts = defaultdict(int)
        self.frame_count = 0
        self.prev_time = time.time()
        
        print("✅ Traffic Monitor ready!")
    
    def process_frame(self, frame):
        """Process single frame: detect → track → count → annotate"""
        
        # Initialize counting line on first frame
        if self.line_zone is None:
            height, width = frame.shape[:2]
            start = sv.Point(0, height // 2)
            end = sv.Point(width, height // 2)
            self.line_zone = sv.LineZone(start=start, end=end)
        
        # AI Detection
        results = self.model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # Filter vehicles only (car=2, motorcycle=3, bus=5, truck=7)
        vehicle_classes = [2, 3, 5, 7]
        vehicle_mask = np.isin(detections.class_id, vehicle_classes)
        detections = detections[vehicle_mask]
        
        # Track vehicles (assign IDs)
        detections = self.tracker.update_with_detections(detections)
        
        # Count crossings
        self.line_zone.trigger(detections=detections)
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - self.prev_time) if self.prev_time else 0
        self.prev_time = current_time
        
        # Draw annotations
        annotated_frame = frame.copy()
        
        # Draw counting line
        self.line_annotator.annotate(annotated_frame, line_counter=self.line_zone)
        
        # Draw boxes
        annotated_frame = self.box_annotator.annotate(
            scene=annotated_frame,
            detections=detections
        )
        
        # Draw labels
        labels = [
            f"#{int(tracker_id)} {self.model.names[class_id]}"
            for tracker_id, class_id in zip(detections.tracker_id, detections.class_id)
        ]
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels
        )
        
        # Add stats overlay
        stats_text = [
            f"FPS: {fps:.1f}",
            f"Active: {len(detections)}",
            f"IN: {self.line_zone.in_count}",
            f"OUT: {self.line_zone.out_count}"
        ]
        
        y_offset = 30
        for text in stats_text:
            cv2.putText(
                annotated_frame, text, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            y_offset += 35
        
        # Return results
        stats = {
            'fps': fps,
            'active_vehicles': len(detections),
            'total_in': self.line_zone.in_count,
            'total_out': self.line_zone.out_count,
            'total_crossed': self.line_zone.in_count + self.line_zone.out_count
        }
        
        self.frame_count += 1
        return annotated_frame, stats


# Quick test
if __name__ == "__main__":
    print("Testing detector...")
    monitor = TrafficMonitor()
    cap = cv2.VideoCapture("traffic.mp4")
    
    if not cap.isOpened():
        print("❌ Cannot open video!")
        exit()
    
    print("✅ Press 'q' to quit")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        annotated_frame, stats = monitor.process_frame(frame)
        
        # Resize for display
        display_frame = cv2.resize(annotated_frame, (1280, 720))
        cv2.imshow("Traffic Monitor", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Test complete!")

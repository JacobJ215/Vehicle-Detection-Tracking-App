import cv2
from ultralytics import YOLO
import supervision as sv
import streamlit as st
import tempfile

# Define start and end points of the counting line
START = sv.Point(0, 850)
END = sv.Point(1920, 850)

def run_yolov8(video_path, confidence, stframe):
    # Load YOLOv8 model
    model = YOLO("yolov8l.pt")
    
    # Initialize a counting line
    line_counter = sv.LineZone(start=START, end=END)
    
    # Initialize annotators for line and bounding boxes
    line_zone_annotator = sv.LineZoneAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=2
    )
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5
    )
    
    # Loop through results from the YOLO model
    for result in model.track(source=video_path, stream=True, tracker="bytetrack.yaml", conf=confidence):
        frame = result.orig_img
        detections = sv.Detections.from_yolov8(result)

        if detections is not None:
            if result.boxes.id is not None:
                detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
            
            # Filter out unwanted classes (class_id 60 and 0)
            detections = detections[(detections.class_id != 60) & (detections.class_id != 0)]

            if detections.confidence.size > 0:
                labels = [
                    f"#{tracker_id}{model.model.names[class_id]} {confidence:0.2f}"
                    for confidence, class_id, tracker_id in zip(detections.confidence, detections.class_id, detections.tracker_id)
                ]
                frame = box_annotator.annotate(
                    scene=frame, 
                    detections=detections, 
                    labels=labels
                )
            # Trigger counting for objects crossing the line
            line_counter.trigger(detections=detections)
            # Annotate frame with counting information
            line_zone_annotator.annotate(frame=frame, line_counter=line_counter)
            # Display the annotated frame
            stframe.image(frame, channels='BGR', use_column_width=True)

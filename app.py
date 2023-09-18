import streamlit as st
import cv2
import numpy as np
import tempfile
from yolonas_sort import run_yolonas
from yolov8_bytetrack import run_yolov8

def main():
    st.title("Vehicle Detection + Tracking App 🏎️")       
    # Sidebar for user input
    model_choice = st.sidebar.radio("Select Model", ("YOLO-NAS", "YOLOv8")) 
    confidence = st.sidebar.slider(
        'Confidence', min_value=0.0, max_value=1.0
    )

    uploaded_file = st.sidebar.file_uploader("Choose a video file:", type=["mp4", "avi", "mov", "asf"])
    if model_choice == "YOLO-NAS":
        st.header("Using YOLO-NAS + SORT")
        with st.spinner():

            # Create a temporary file to store the uploaded video
            tffile = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            stframe = st.empty()

            # If no video is uploaded, display the demo video in the sidebar
            if not uploaded_file:
                demo = "./videos/video0.mp4"
                tffile.name = demo
                demo_vid = open(tffile.name, 'rb')
                demo_bytes = demo_vid.read()
                st.sidebar.text("Input Video")
                st.sidebar.video(demo_bytes)
            else:
                # Write the uploaded video to the temporary file
                tffile.write(uploaded_file.read())
                vid = open(tffile.name, 'rb')
                vid_bytes = vid.read()
                st.sidebar.text("Input Video")
                st.sidebar.video(vid_bytes)

            # Create a button to start inference
            if st.sidebar.button("Start Inference"):
                # Process video using the detection and tracking module
                run_yolonas(tffile.name, confidence, stframe)
                st.write("Video processing complete.")

    elif model_choice == "YOLOv8":
        st.header("Using YOLOv8 + ByteTrack + Supervision")
        with st.spinner():
            # Create a temporary file to store the uploaded video
            tffile = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            stframe = st.empty()

            # If no video is uploaded, display the demo video in the sidebar
            if not uploaded_file:
                demo = "./videos/video0.mp4"
                tffile.name = demo
                demo_vid = open(tffile.name, 'rb')
                demo_bytes = demo_vid.read()
                st.sidebar.text("Input Video")
                st.sidebar.video(demo_bytes)
            else:
                # Write the uploaded video to the temporary file
                tffile.write(uploaded_file.read())
                vid = open(tffile.name, 'rb')
                vid_bytes = vid.read()
                st.sidebar.text("Input Video")
                st.sidebar.video(vid_bytes)

            # Create a button to start inference
            if st.sidebar.button("Start Inference"):
                # Process video using the detection and tracking module
                run_yolov8(tffile.name, confidence, stframe)
                st.write("Video processing complete.")



if __name__ == '__main__':
    main()
# Vehicle Detection + Tracking App
![](images/vehicle_tracking.gif)

[GitHub](https://github.com/JacobJ215/Vehicle-Detection-Tracking-App)

## Overview
This repository contains a Streamlit web application for vehicle tracking using different SOTA object detection models. The app offers two options: YOLO-NAS with SORT tracking and YOLOv8 with ByteTrack and Supervision tracking. It enables users to upload a video file, set confidence levels, and visualize the tracking results in real-time. 

## Technologies Used
- Python
- Streamlit
- OpenCV
- PyTorch
- YOLO-NAS
- YOLOv8
- SORT 
- ByteTrack
- Supervision 

## How to Clone and Run the App
1. **Clone the Repository**

    ```
    git clone https://github.com/your-username/vehicle-tracking-app.git
    ```

2. **Install Dependencies**

    ```
    pip install -r requirements.txt
    ```


3. **Run the App**

    ```
    streamlit run app.py
    ```



## Object Detection and Tracking Overview

### YOLO-NAS with SORT Tracking
- YOLO (You Only Look Once) is a real-time object detection system. YOLO-NAS (Neural Architecture Search) is a YOLO variant that is optimized for detecting a wide range of objects.
- SORT (Simple Online and Realtime Tracking) is a simple, yet effective, online multiple object tracking algorithm. It associates detections in consecutive frames to track objects.

### YOLOv8 with ByteTrack and Supervision
- YOLOv8 is another variant of YOLO, known for its high accuracy and speed in object detection.
- ByteTrack is a state-of-the-art online multi-object tracking algorithm that employs motion and appearance cues to track objects effectively.
- Supervision is a toolkit for YOLO-based object detection models.

## YOLO-NAS Overview
![](./images/yolo_nas_frontier.png)
- YOLO-NAS, short for You Only Look Once with Neural Architecture Search, is a cutting-edge object detection model optimized for both accuracy and low-latency inference.
- Developed by Deci, YOLO-NAS employs state-of-the-art techniques like Quantization Aware Blocks and selective quantization for superior performance.
- It sets a new standard for state-of-the-art (SOTA) object detection, making it an ideal choice for a wide range of applications including autonomous vehicles, robotics, and video analytics.

### Unique Features of YOLO-NAS
- Utilizes Quantization Aware Blocks for efficient inference without sacrificing accuracy.
- Incorporates AutoNAC technology for optimal architecture design, balancing accuracy, speed, and complexity.
- Supports INT8 quantization for unprecedented runtime performance.
- Pre-trained weights available for research use on SuperGradients, Deci’s PyTorch-based computer vision training library.

## YOLOv8 Overview
![](./images/yolo-comparison-plots.png)
- YOLOv8 is a state-of-the-art object detection model developed by Ultralytics, known for its high accuracy and developer-friendly features.
- It introduces significant architectural improvements over its predecessors and is actively supported by the community.

### Main Features of YOLOv8
- Utilizes an anchor-free detection system for more accurate predictions.
- Incorporates new convolutional blocks for improved performance.
- Implements Mosaic Augmentation for enhanced training.


## ByteTrack and SORT Tracking
- ByteTrack and SORT are advanced online multi-object tracking algorithms that complement YOLO-based object detection models.
- ByteTrack leverages motion and appearance cues for effective object tracking.
- SORT associates detections in consecutive frames to create object tracks.





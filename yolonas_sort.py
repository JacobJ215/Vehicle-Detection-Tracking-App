import cv2
import torch
import math
import numpy as np
from super_gradients.training import models
from sort import Sort

def run_yolonas(video_path, confidence, stframe):
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    # Set device for model (GPU if available, else CPU)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # Load YOLO model
    model = models.get('yolo_nas_m', pretrained_weights="coco").to(device)

    # List of object class names
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
            ]

    # Initialize SORT tracker
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    totalcountup = []    # List to store IDs of objects entering
    totalcountdown = []  # List to store IDs of objects leaving

    # Define coordinates for boundary lines
    limitdown = [0, 850, 900, 850]  # [x1, y1, x2, y2] for lower boundary
    limitup = [950, 850, 1920, 850]  # [x1, y1, x2, y2] for upper boundary

    while True:
        ret, frame = cap.read()
        if ret:
            detections = np.empty((0, 5))
            result = list(model.predict(frame, conf=0.35))[0]
            bbox_xyxys = result.prediction.bboxes_xyxy.tolist()
            confidences = result.prediction.confidence
            labels = result.prediction.labels.tolist()
            for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidences, labels):
                bbox = np.array(bbox_xyxy)
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                classname = int(cls)
                class_name = classNames[classname]
                conf = math.ceil((confidence*100))/100
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
            resultsTracker = tracker.update(detections)
            cv2.line(frame, (limitup[0], limitup[1]), (limitup[2], limitup[3]), (255, 255,255),5)
            cv2.line(frame, (limitdown[0], limitdown[1]), (limitdown[2], limitdown[3]), (255, 255, 255), 5)
            for result in resultsTracker:
                x1, y1, x2, y2, id = result
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255,144, 30), 3)
                label = f'#{int(id)}{class_name} {conf}'
                t_size = cv2.getTextSize(label, 0, fontScale = 1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] -3
                cv2.rectangle(frame, (x1, y1), c2, [255,144, 30], -1, cv2.LINE_AA)
                cv2.putText(frame, label, (x1, y1-2), 0, 1, [255, 255, 255], thickness=1, lineType = cv2.LINE_AA)
                if limitup[0] < cx < limitup[2] and limitup[1] -15 < cy < limitup[3] + 15:
                    if totalcountup.count(id) == 0:
                        totalcountup.append(id)
                        cv2.line(frame, (limitup[0], limitup[1]), (limitup[2], limitup[3]), (0, 255, 0), 5)
                if limitdown[0] < cx < limitdown[2] and limitdown[1] -15 < cy < limitdown[3] + 15:
                    if totalcountdown.count(id) == 0:
                        totalcountdown.append(id)
                        cv2.line(frame, (limitdown[0], limitdown[1]), (limitdown[2], limitdown[3]), (0, 255, 0), 5)
            cv2.rectangle(frame, (1267, 65), (1617, 97), [255,144, 30], -1, cv2.LINE_AA)
            cv2.putText(frame, str("Vehicle Entering") + ":" + str(len(totalcountup)), (1317, 91), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
            cv2.rectangle(frame, (100, 65), (441, 97), [255,144, 30], -1, cv2.LINE_AA)
            cv2.putText(frame, str("Vehicle Leaving") + ":" + str(len(totalcountdown)), (141, 91), 0, 1, [225, 255, 255],thickness=2, lineType=cv2.LINE_AA)
            stframe.image(frame, channels='BGR', use_column_width=True)
        else:
            break

    # Release video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

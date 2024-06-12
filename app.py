import cv2
import streamlit as st
import pandas as pd
from ultralytics import YOLO
from tracker import *
import numpy as np
from PIL import Image

# Load the YOLOv8 model
model = YOLO('yolov8l.pt')

# Load pre-trained face and gender classification models
face_net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')
gender_list = ['Male', 'Female']

# Define color ranges for red and blue (BGR format)
red_lower = np.array([0, 0, 100])
red_upper = np.array([50, 50, 255])
blue_lower = np.array([100, 0, 0])
blue_upper = np.array([255, 50, 50])

def detect_color(image, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    car_region = image[y1:y2, x1:x2]
    mask_red = cv2.inRange(car_region, red_lower, red_upper)
    mask_blue = cv2.inRange(car_region, blue_lower, blue_upper)

    red_area = cv2.countNonZero(mask_red)
    blue_area = cv2.countNonZero(mask_blue)

    if red_area > blue_area:
        return 'red'
    elif blue_area > red_area:
        return 'blue'
    else:
        return 'unknown'

def classify_gender(face_image):
    blob = cv2.dnn.blobFromImage(face_image, 1.0, (227, 227), (104.0, 177.0, 123.0))
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    return gender_list[gender_preds[0].argmax()]

st.title("Object and Gender Count from Video")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    cap = cv2.VideoCapture(uploaded_file.name)

    with open("coco.txt", "r") as f:
        class_list = f.read().strip().split("\n")

    tracker = Tracker()

    # Define the line for counting
    line_y = 250
    offset = 5

    # Initialize counts for different categories
    category_counts = {
        "car": 0,
        "truck": 0,
        "bike": 0,
        "people": 0,
        "red_car": 0,
        "blue_car": 0,
        "male": 0,
        "female": 0
    }

    # Resize dimensions
    resize_width = 640
    resize_height = 360

    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (resize_width, resize_height))

        results = model.predict(frame)
        detections = results[0].boxes.data.cpu().numpy()  # Convert to numpy array

        detection_list = []
        for det in detections:
            x1, y1, x2, y2, conf, class_id = det
            class_id = int(class_id)
            class_name = class_list[class_id]
            if class_name in ['car', 'truck', 'bike', 'person']:
                detection_list.append([int(x1), int(y1), int(x2), int(y2), class_name])

        bbox_id = tracker.update(detection_list)
        for bbox in bbox_id:
            x3, y3, x4, y4, object_id = bbox
            cx = (x3 + x4) // 2
            cy = (y3 + y4) // 2

            # Check if the object crosses the line
            if line_y - offset < cy < line_y + offset:
                if 'car' in class_name:
                    category_counts["car"] += 1
                    car_color = detect_color(frame, (x3, y3, x4, y4))
                    if car_color == 'red':
                        category_counts["red_car"] += 1
                    elif car_color == 'blue':
                        category_counts["blue_car"] += 1
                elif 'truck' in class_name:
                    category_counts["truck"] += 1
                elif 'bike' in class_name:
                    category_counts["bike"] += 1
                elif 'person' in class_name:
                    category_counts["people"] += 1

                    # Face detection
                    face_blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
                    face_net.setInput(face_blob)
                    detections = face_net.forward()

                    for i in range(detections.shape[2]):
                        confidence = detections[0, 0, i, 2]
                        if confidence > 0.5:
                            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                            (startX, startY, endX, endY) = box.astype("int")
                            face = frame[startY:endY, startX:endX]
                            if face.shape[0] > 0 and face.shape[1] > 0:
                                gender = classify_gender(face)
                                if gender == 'Male':
                                    category_counts["male"] += 1
                                elif gender == "Female":
                                    category_counts["female"] += 1

            # Draw bounding box and class name
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (255, 255, 255), 2)

        # Display category counts
        y_offset = 30
        for category, count in category_counts.items():
            cv2.putText(frame, f"{category}: {count}", (10, y_offset), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
            y_offset += 40
            

        stframe.image(frame, channels="BGR")
        

    cap.release()

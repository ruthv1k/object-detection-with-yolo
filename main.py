"""
Author: Ruthvik

Description: This source code implements real-time object detection using the YOLO (You Only Look Once) algorithm.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import threading
import time

root = tk.Tk()
root.geometry("800x600")
root.title("Object Detection")

capture = cv2.VideoCapture(0)
yolo = cv2.dnn.readNetFromDarknet("model/yolov3.cfg", "model/yolov3.weights")

frame = tk.Frame(root)
frame.pack()

image_label = tk.Label(frame)
image_label.pack()

video_label = tk.Label(frame)
video_label.pack()

real_time_video_label = tk.Label(frame)
real_time_video_label.pack()

with open("classes.txt") as f:
    classes = f.read().splitlines()
    f.close()

# Real time object detection using camera


def detect_objects():
    try:
        while True:
            _, captured_frame = capture.read()
            if captured_frame is None:
                break

            height, width, _ = captured_frame.shape

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

            blob = cv2.dnn.blobFromImage(
                captured_frame, 1 / 255.0, (416, 416), swapRB=True, crop=False
            )
            yolo.setInput(blob)
            layer_outputs = yolo.forward(yolo.getUnconnectedOutLayersNames())

            boxes = []
            confidences = []
            class_ids = []

            for output in layer_outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])
                        class_ids.append(class_id)

            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            for i in indices:
                box = boxes[i]
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]

                cv2.rectangle(captured_frame, (x, y),
                              (x + w, y + h), (0, 255, 0), 2)

                if class_ids[i] < len(classes):
                    label = str(classes[class_ids[i]])
                else:
                    label = "Unknown"

                cv2.putText(
                    captured_frame,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            captured_frame = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB)
            captured_frame = Image.fromarray(captured_frame)
            captured_frame = ImageTk.PhotoImage(captured_frame)
            real_time_video_label.config(image=captured_frame)
            real_time_video_label.image = captured_frame

    except Exception as e:
        print(e)

# Object detection inside an image


def upload_image():
    try:
        file_path = filedialog.askopenfilename()

        image = cv2.imread(file_path)
        image = cv2.resize(image, (416, 416))

        height, width, _ = image.shape
        blob = cv2.dnn.blobFromImage(
            image, 1 / 255.0, (416, 416), swapRB=True, crop=False
        )
        yolo.setInput(blob)
        layer_outputs = yolo.forward(yolo.getUnconnectedOutLayersNames())

        boxes = []
        confidences = []
        class_ids = []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in indices:
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if class_ids[i] < len(classes):
                label = str(classes[class_ids[i]])
            else:
                label = "Unknown"

            cv2.putText(
                image,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        image_label.config(image=image)
        image_label.image = image

    except Exception as e:
        print(e)

# Object detection in a video


def upload_video():
    try:
        file_path = filedialog.askopenfilename()

        capture = cv2.VideoCapture(file_path)

        frame_rate = capture.get(cv2.CAP_PROP_FPS)

        while True:
            _, frame = capture.read()
            frame = cv2.resize(frame, (416, 416))

            if frame is None:
                break

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

            height, width, channels = frame.shape
            blob = cv2.dnn.blobFromImage(
                frame, 1 / 255.0, (416, 416), swapRB=True, crop=False
            )
            yolo.setInput(blob)
            layer_outputs = yolo.forward(yolo.getUnconnectedOutLayersNames())

            boxes = []
            confidences = []
            class_ids = []

            for output in layer_outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])
                        class_ids.append(class_id)

            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            for i in indices:
                box = boxes[i]
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if class_ids[i] < len(classes):
                    label = str(classes[class_ids[i]])
                else:
                    label = "Unknown"

                cv2.putText(
                    frame,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = ImageTk.PhotoImage(frame)
            video_label.config(image=frame)
            video_label.image = frame

            root.update()

            time.sleep(1 / frame_rate)

    except Exception as e:
        print(e)

# Uses threading during real time object detection for improved performance


def start_detection():
    detection_thread = threading.Thread(target=detect_objects)

    detection_thread.start()


image_upload_button = tk.Button(
    frame, text="Upload Image", command=upload_image)
image_upload_button.pack(pady=16)

video_upload_button = tk.Button(
    frame, text="Upload Video", command=upload_video)
video_upload_button.pack(pady=16)

button = tk.Button(frame, text="Start Detection", command=start_detection)
button.pack(pady=16)

root.mainloop()

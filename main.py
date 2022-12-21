"""
Author: Ruthvik

Description: This source code implements real-time object detection using the YOLO (You Only Look Once) algorithm.
"""

import threading
import time
import tkinter as tk
from tkinter import filedialog, Toplevel

import cv2
import numpy as np
from PIL import Image, ImageTk

from helpers import (draw_bounding_boxes,
                     get_confidence_and_detection_coordinates)

root = tk.Tk()
root.geometry("400x300")
root.title("Object Detection")

capture = cv2.VideoCapture(0)
yolo = cv2.dnn.readNetFromDarknet("model/yolov3.cfg", "model/yolov3.weights")

frame = tk.Frame(root)
frame.pack()

# Real time object detection using camera


def detect_objects():
    try:
        detect_objects_window = Toplevel(root)
        detect_objects_window.title("Real Time Object Detection Using Camera")
        detect_objects_window.geometry("600x500")
        detect_objects_window.attributes('-topmost', True)

        real_time_video_label = tk.Label(detect_objects_window)
        real_time_video_label.pack()

        while True:
            _, captured_frame = capture.read()
            if captured_frame is None:
                break

            height, width, _ = captured_frame.shape

            blob = cv2.dnn.blobFromImage(
                captured_frame, 1 / 255.0, (416, 416), swapRB=True, crop=False
            )
            yolo.setInput(blob)
            layer_outputs = yolo.forward(yolo.getUnconnectedOutLayersNames())

            boxes, confidences, class_ids = get_confidence_and_detection_coordinates(
                layer_outputs, width, height)
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            draw_bounding_boxes(captured_frame, indices, boxes, class_ids)

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
        image_window = Toplevel(root)
        image_window.title("Object Detection In An Image")
        image_window.geometry("600x500")

        image_label = tk.Label(image_window)
        image_label.pack()

        file_path = filedialog.askopenfilename()

        if file_path:
            image_window.attributes('-topmost', True)

        image = cv2.imread(file_path)
        image = cv2.resize(image, (416, 416))

        height, width, _ = image.shape
        blob = cv2.dnn.blobFromImage(
            image, 1 / 255.0, (416, 416), swapRB=True, crop=False
        )
        yolo.setInput(blob)
        layer_outputs = yolo.forward(yolo.getUnconnectedOutLayersNames())

        boxes, confidences, class_ids = get_confidence_and_detection_coordinates(
            layer_outputs, width, height)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        draw_bounding_boxes(image, indices, boxes, class_ids)

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
        video_window = Toplevel(root)
        video_window.title("Object Detection In A Video")
        video_window.geometry("600x500")

        video_label = tk.Label(video_window)
        video_label.pack()

        file_path = filedialog.askopenfilename()

        if file_path:
            video_window.attributes('-topmost', True)

        capture = cv2.VideoCapture(file_path)

        frame_rate = capture.get(cv2.CAP_PROP_FPS)

        while True:
            _, captured_frame = capture.read()
            captured_frame = cv2.resize(captured_frame, (416, 416))

            if captured_frame is None:
                break

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

            height, width, _ = captured_frame.shape
            blob = cv2.dnn.blobFromImage(
                captured_frame, 1 / 255.0, (416, 416), swapRB=True, crop=False
            )
            yolo.setInput(blob)
            layer_outputs = yolo.forward(yolo.getUnconnectedOutLayersNames())

            boxes, confidences, class_ids = get_confidence_and_detection_coordinates(
                layer_outputs, width, height)

            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            draw_bounding_boxes(captured_frame, indices, boxes, class_ids)

            captured_frame = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB)
            captured_frame = Image.fromarray(captured_frame)
            captured_frame = ImageTk.PhotoImage(captured_frame)
            video_label.config(image=captured_frame)
            video_label.image = captured_frame

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

real_time_object_detection_button = tk.Button(
    frame, text="Start Camera", command=start_detection)
real_time_object_detection_button.pack(pady=16)

root.mainloop()

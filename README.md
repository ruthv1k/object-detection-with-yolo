# Object Detection with YOLO

A minor project on real time object detection using YOLO algorithm

## Dependencies Required

-   OpenCV
-   NumPy
-   Tkinter
-   Pillow

## Setup

-   Make sure you have python > 3.7 installed
-   pip install -r requirements.txt
-   Download the config and weight files from the links below and place them inside `model/` folder and name them as `yolov3.cfg` and `yolov3.weights` respectively
-   python main.py

## Model Info

-   Model: YOLOv3-416
-   Config: https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
-   Weights: https://pjreddie.com/media/files/yolov3.weights

## Screenshots
![image](https://user-images.githubusercontent.com/91245898/208859393-9813086c-a9c2-43a4-b6d0-cdacfb50bde1.png)

## Known Issues

-   GUI Window doesn't close after clicking any key / Esc key
-   Close objects or small objects detection may not be accurate
-   Limited number of classes (80) are identified

## Improvements to be made

-   Bounding boxes should be coloured differently for different classes identified

# References

-   https://github.com/pjreddie/darknet
-   https://pjreddie.com/darknet/yolo/

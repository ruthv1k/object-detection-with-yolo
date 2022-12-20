# Object Detection with YOLO

A minor project on real time object detection using YOLO algorithm

## Dependencies Required

-   OpenCV
-   NumPy
-   Tkinter
-   Pillow

## Setup

-   pip install -r requirements.txt
-   python main.py

## Model Info

-   Model: YOLOv3-416
-   Config: https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
-   Weights: https://pjreddie.com/media/files/yolov3.weights

## Known Issues

-   Window has to be closed after clicking any key or Esc key
-   Close objects or small objects detection may not be accurate
-   Limited number of classes are identified

## Improvements to be made

-   A seperate window has to be opened to show the objects detected
-   Bounding boxes should be coloured differently for different classes identified

# References

-   https://github.com/pjreddie/darknet
-   https://pjreddie.com/darknet/yolo/

import cv2
import numpy as np


def draw_bounding_boxes(frame, indices, boxes, class_ids):
    with open("classes.txt") as f:
        classes = f.read().splitlines()
        f.close()

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


def get_confidence_and_detection_coordinates(layer_outputs, width, height):
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

    return [boxes, confidences, class_ids]

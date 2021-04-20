import cv2
import numpy as np
import os
import random
import imutils
import time
import glob


from nms import nms

class YOLO_Detector:
    def __init__(self, label_list, model_path, cofig_path):
        self.classes = label_list
        # LOad model
        self.net = cv2.dnn.readNet(model_path, cofig_path)
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def predict(self, img, ID):
        height, width, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 0.00392, (832, 832), (0, 0, 0), True, crop=False)

        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)  


        boxes, confidences = nms(boxes,confidences,0.4)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                color = self.colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, label, (x, y-2), font, 1, color, 2)
        cv2.imwrite('./output/images/' + str(ID) + '.jpg', img)


        return class_ids, boxes, confidences
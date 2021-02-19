import os
import cv2
import json
import random
import itertools
import numpy as np
import argparse
import cv2

from time import gmtime, strftime

def predict(image, predictor, list_labels):
    outputs = predictor(image)

    boxes = outputs['instances'].pred_boxes
    scores = outputs['instances'].scores
    classes = outputs['instances'].pred_classes

    list_boxes = []
    # list_paths = []
    # list_vehicles = []
    list_scores = []
    list_classes = []

    for i in range(len(classes)):
        if (scores[i] > 0.6):
            for j in boxes[i]:
                x1 = int(j[0])
                y1 = int(j[1])
                x2 = int(j[2]) 
                y2 = int(j[3]) 

            print("min: ", (x1, y1))
            print("max: ", (x2, y2))

            score = float(scores[i])
            class_id = list_labels[int(classes[i])]

            list_boxes.append([x1, y1, x2, y2])
            list_scores.append(score)
            list_classes.append(class_id)

    return list_boxes, list_scores, list_classes

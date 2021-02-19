
import cv2
import numpy as np
import time
import logging
import traceback
import os
import io
import requests
import random
import json

from utils.utils import load_class_names
from utils.parser import get_config
from utils.draw import draw_bbox

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg as config_detectron

from src.predict import predict

# setup config
cfg = get_config()
cfg.merge_from_file('configs/service.yaml')
cfg.merge_from_file('configs/rcode.yaml')

# create log_file, rcode
LOG_PATH = cfg.SERVICE.LOG_PATH
RCODE = cfg.RCODE

if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_PATH)

# setup host, port
HOST = cfg.SERVICE.SERVICE_IP
PORT = cfg.SERVICE.PORT

# set up detectron
path_weigth = cfg.SERVICE.DETECT_WEIGHT
path_config = cfg.SERVICE.DETECT_CONFIG
confidences_threshold = cfg.SERVICE.THRESHOLD
num_of_class = cfg.SERVICE.NUMBER_CLASS

detectron = config_detectron()
detectron.MODEL.DEVICE = cfg.SERVICE.DEVICE
detectron.merge_from_file(path_config)
detectron.MODEL.WEIGHTS = path_weigth

detectron.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidences_threshold
detectron.MODEL.ROI_HEADS.NUM_CLASSES = num_of_class

PREDICTOR = DefaultPredictor(detectron)

# create labels
CLASSES = load_class_names(cfg.SERVICE.CLASSES)

image = cv2.imread('images/test.jpg')

height, width, channels = image.shape
center_image = (width//2, height//2)
print("shape image: ", (width, height))
list_boxes, list_scores, list_classes = predict(
    image, PREDICTOR, CLASSES)
print('list_boxes', list_boxes)
print('list_classes', list_classes)

# draw
# image = draw_bbox(image, list_boxes, list_scores, list_classes)
# cv2.imwrite("image.jpg", image)

i = 0
len_boxes = len(list_boxes)
point_tl = None
point_tr = None
point_bl = None
point_br = None
receipt = None
while i < len_boxes:
    bbox = list_boxes[i]
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]
    w = x2 - x1
    h = y2 - y1
    center_x = x1 + w//2
    center_y = y1 + h//2
    center = (center_x, center_y)
    # print("max: ", (x1, y1))
    # print("min: ", (x2, y2))
    if list_classes[i] == 'top_right':
        point_tr = center
    elif list_classes[i] == 'bottom_left':
        point_bl = center
    elif list_classes[i] == 'bottom_right':
        point_br = center
    elif list_classes[i] == 'top_left':
        point_tl = center
    elif list_classes[i] == 'receipt':
        receipt = bbox

    i += 1

result = {'point_tl': point_tl, 'point_tr': point_tr,
            'point_bl': point_bl, 'point_br': point_br, 'receipt': receipt}
import cv2
import numpy as np 
import time
import os
import io
import requests
import random
import json

# pannet
# from receiptextraction.pannet import Pytorch_model
# import receiptextraction.pannet
# vietocr
from task2.model.predict import load_images_to_predict
# extract info
from e2e.mc_ocr_rivf2020.post_processing.extract_info import extract_info, add_info_compare
# detec receipt 
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg as config_detectron
from detect_receipt_api.src.predict import predict
from detect_receipt_api.utils.utils import load_class_names
from detect_receipt_api.utils.parser import get_config
from detect_receipt_api.utils.draw import draw_bbox
# rotated receipt 90 180
from timm.models import create_model, apply_test_time_pool
from timm.data import Dataset, create_loader, resolve_data_config
from timm.utils import AverageMeter, setup_default_logging
from rotated_receipt_90_180.load_model import *
# rotated receipt 0 <- 45
from e2e.mc_ocr_rivf2020.pre_processing.rotated_receipt_khanh import load_bbox, crop_text
# visualize
from visualize import *

#img_path = 'private/mcocr_private_145121pzatx.jpg'

# model_pannet = Pytorch_model("receiptextraction/PANNet_model_pretrain_SOIRE.pth", gpu_id=0)


# setup config detect receipt 
cfg = get_config()
cfg.merge_from_file('detect_receipt_api/configs/service.yaml')
cfg.merge_from_file('detect_receipt_api/configs/rcode.yaml')

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

#


OUTPUT_TXT = 'output_pipeline/output_txt/'
# OUTPUT_PATH_POST = 'output_pipeline/output'
OUTPUT_PATH_POST = '/output'
OUTPUT_PATH_CROP_RECEIPT = 'output_pipeline/output_crop/'
OUTPUT_PATH_ROTATED_45 = '/output/output_rotated_45/'
OUTPUT_PATH_ROTATED_90_180 = 'output_pipeline/output_rotated_90_180/'
# INPUT_FOLDER_IMAGES = 'mcocr_private_test_data'
INPUT_FOLDER_IMAGES = 'input'
# TASK1_URL = 'http://service.aiclub.cs.uit.edu.vn/gpu150/pannet/predict'
TASK1_URL = 'http://0.0.0.0:5010/predict'

if not os.path.exists(OUTPUT_TXT):
    os.makedirs(OUTPUT_TXT)
if not os.path.exists(OUTPUT_PATH_POST):
    os.makedirs(OUTPUT_PATH_POST)
if not os.path.exists(OUTPUT_PATH_CROP_RECEIPT):
    os.makedirs(OUTPUT_PATH_CROP_RECEIPT)
if not os.path.exists(OUTPUT_PATH_ROTATED_45):
    os.makedirs(OUTPUT_PATH_ROTATED_45)
if not os.path.exists(OUTPUT_PATH_ROTATED_90_180):
    os.makedirs(OUTPUT_PATH_ROTATED_90_180)



def pannet_json(boxes_list):
    result = []
    cnt = 0 
    
    while cnt < len(boxes_list):
        # print(boxes_list[cnt])
        # print(type(boxes_list[cnt]))
        box_list = boxes_list[cnt].ravel().tolist()
        box_int_list = [int(i) for i in box_list]
        my_dict = {
            "bbox": box_int_list
        }
        result.append(my_dict)
        cnt += 1
    
    return {'result': result}

def vietocr_json(words_list, prob_list):
    result = []
    cnt = 0 
    
    while cnt < len(words_list):
        words = words_list[cnt]
        prob  = prob_list[cnt]
        my_dict = {
            "words": words,
            "pros": prob
        }
        result.append(my_dict)
        cnt += 1
    return {'result': result}

def detec_receipt_json(list_boxes, list_classes, img):
    for cl, box in zip(list_classes,list_boxes):
        if cl == 'receipt':
            crop = img[box[1]:box[3], box[0]:box[2]]
            return crop
    return img

def output_txt(data_task1, data_task2, output_path):
    bboxs = data_task1['result']
    words = data_task2['result']
    # print(words)
    result_txt_list = ''
    for box, word in zip(bboxs, words):
        result_txt_line = ''
        box = box['bbox']
        word = word['words']
        # print(box)
        # print(word)
        for b in box:
            result_txt_line += str(b) + ' '
            # print(result_txt_line)
        result_txt_line += str(word)
        result_txt_list += result_txt_line + '\n'
    with open(output_path, 'w') as out:
        out.write(result_txt_list)
        print('output_path: ',output_path, '----ok----')

if __name__ == '__main__':
    # image = cv2.imread(img_path)
    # img_name = img_path.split('/')[-1]
    # output_txt_path = os.path.join(OUTPUT_TXT, img_name.replace('jpg', 'txt'))
    # preds, boxes_list, t = model_pannet.predict(image)
    # detect_pannet = pannet_json(boxes_list)
    # print(detect_pannet)
    # words_list, prob_list = load_images_to_predict(detect_pannet, image)
    # detect_vietocr = vietocr_json(words_list, prob_list)
    # print(detect_vietocr)
    # output_txt(detect_pannet, detect_vietocr, output_txt_path)

    # input_folder_img = 'input_img_test'

    # result_extract_info_txt = extract_info(input_folder_img, OUTPUT_TXT, OUTPUT_PATH_POST, 'json_visualize_path')

    path, dirs, files = next(os.walk(INPUT_FOLDER_IMAGES))
    print(len(files))
    urls = []
    time_pannet = 0
    time_xoay_receipt = 0
    time_xoay_90_180 = 0
    time_viet_ocr = 0 
    time_detec_reciept = 0
    time_extract_info = 0
    t = time.time()
    for fn in files:
        print('------------read image:',fn)
        img_path = os.path.join(path, fn)
        image = cv2.imread(img_path)
        img_name = img_path.split('/')[-1]
        output_txt_path = os.path.join(OUTPUT_TXT, img_name.replace('jpg', 'txt'))

        print('--------------CROP_IMG_DETECT_RECEIPT----------------')
        t1 = time.time()
        height, width, channels = image.shape
        center_image = (width//2, height//2)
        # print("shape image: ", (width, height))
        list_boxes, list_scores, list_classes = predict(
            image, PREDICTOR, CLASSES)
        time_detec_reciept += time.time() - t1 
        # print('list_boxes', list_boxes)
        # print('list_classes', list_classes)
        image = detec_receipt_json(list_boxes, list_classes, image)
        cv2.imwrite(OUTPUT_PATH_CROP_RECEIPT + fn, image)

        print('--------------ROTATED_RECEIPT_90_180----------------')
        img = cv2.resize(image, (600,600))
        t2 = time.time()
        rotated_func = rotate_img(img)
        print(rotated_func)
        rotated_func = rotated_func.split('\n')
        time_xoay_90_180 += time.time() - t2 
        if rotated_func[0] != 'None' and float(rotated_func[1]) > 0.4 and rotated_func[0] != 'ROTATE_180':
            dic_rotate_fuc = {'ROTATE_90_CLOCKWISE':cv2.ROTATE_90_CLOCKWISE, 'ROTATE_90_COUNTERCLOCKWISE':cv2.ROTATE_90_COUNTERCLOCKWISE, 'ROTATE_180':cv2.ROTATE_180}
            image = cv2.rotate(image, dic_rotate_fuc[rotated_func[0]])
        if rotated_func[0] == 'ROTATE_180' and float(rotated_func[1]) > 0.56:
            dic_rotate_fuc = {'ROTATE_90_CLOCKWISE':cv2.ROTATE_90_CLOCKWISE, 'ROTATE_90_COUNTERCLOCKWISE':cv2.ROTATE_90_COUNTERCLOCKWISE, 'ROTATE_180':cv2.ROTATE_180}
            image = cv2.rotate(image, dic_rotate_fuc[rotated_func[0]])
        cv2.imwrite(OUTPUT_PATH_ROTATED_90_180 + fn, image)

        print('--------------DETECT_TEXT_PANNET----------------')
        # preds, boxes_list, t = model_pannet.predict(image)
        # detect_pannet = pannet_json(boxes_list)
        # print(detect_pannet)
        t3 = time.time()
        detect_pannet = requests.post(TASK1_URL, files={"file": (
        "filename", open(OUTPUT_PATH_ROTATED_90_180 + fn, "rb"), "image/jpeg")}).json()
        # print(detect_pannet)
        time_pannet += time.time() -t3

        print('--------------ROTATED_RECEIPT_KHANH----------------')
        t4 = time.time()
        pts = load_bbox(detect_pannet)
        img_bbox = cv2.polylines(image.copy(), pts, True, (0, 0, 255), thickness=2)
        img_bbox = cv2.resize(img_bbox, (img_bbox.shape[1] * 416 // img_bbox.shape[0], 416))
        image = crop_text(image, pts)
        time_xoay_receipt += time.time() - t4
        cv2.imwrite(OUTPUT_PATH_ROTATED_45 + fn, image)

        print('--------------DETECT_TEXT_PANNET----------------')
        # preds, boxes_list, t = model_pannet.predict(image)
        # detect_pannet = pannet_json(boxes_list)
        # print(detect_pannet)
        t5 = time.time()
        detect_pannet = requests.post(TASK1_URL, files={"file": (
        "filename", open(OUTPUT_PATH_ROTATED_45 + fn, "rb"), "image/jpeg")}).json()
        # print(detect_pannet)
        time_pannet += time.time() -t5

        print('--------------REG_TEXT_VIETOCR----------------')
        t6 = time.time()
        words_list, prob_list = load_images_to_predict(detect_pannet, image)
        detect_vietocr = vietocr_json(words_list, prob_list)
        time_viet_ocr += time.time() -t6
        # print(detect_vietocr)
        output_txt(detect_pannet, detect_vietocr, output_txt_path)

        input_folder_img = INPUT_FOLDER_IMAGES

    t7 = time.time()
    result_extract_info_txt = extract_info(input_folder_img, OUTPUT_TXT, OUTPUT_PATH_POST, 'json_visualize_path')
    time_extract_info = time.time() - t7
    full = time.time() - t
    # time_pannet = 0
    # time_xoay_receipt = 0
    # time_xoay_90_180 = 0
    # time_viet_ocr = 0 
    # time_detec_reciept = 0
    # time_extract_info = 0
    print('time_pannet:',time_pannet)
    print('time_xoay_receipt:', time_xoay_receipt)
    print('time_xoay_90_180: ', time_xoay_90_180)
    print('time_viet_ocr: ', time_viet_ocr)
    print('time_detec_reciept: ', time_detec_reciept)
    print('time_extract_info:', time_extract_info)
    print('full: ',full)

import cv2
import numpy as np 
import time
import os
import io
import requests
import random
import json

# pannet
from api_detec_reg.task1.predict import Pytorch_model
import api_detec_reg.task1.predict
# vietocr
from task2.model.predict import load_images_to_predict
# visualize
from visualize import *

img_path = 'private/mcocr_private_145120azxzl.jpg'

model_pannet = Pytorch_model("api_detec_reg/task1/PANNet_model_pretrain_SOIRE.pth", gpu_id=0)

OUTPUT_TXT = 'output_pipeline/output_txt/'
OUTPUT_PATH_POST = 'output_pipeline/output'
OUTPUT_PATH_CROP_RECEIPT = 'output_pipeline/output_crop/'
OUTPUT_PATH_ROTATED_45 = 'output_pipeline/output_rotated_45/'
OUTPUT_PATH_ROTATED_90_180 = 'output_pipeline/output_rotated_90_180/'
INPUT_FOLDER_IMAGES = 'test_img'

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
    image = cv2.imread(img_path)
    img_name = img_path.split('/')[-1]
    output_txt_path = os.path.join(OUTPUT_TXT, img_name.replace('jpg', 'txt'))

    preds, boxes_list, t = model_pannet.predict(image)
    detect_pannet = pannet_json(boxes_list)
    print(detect_pannet)
    # http://service.mmlab.uit.edu.vn/receipt/task1/predict
    # http://service.aiclub.cs.uit.edu.vn/gpu150/pannet/predict
    # detect_pannet = requests.post('http://service.aiclub.cs.uit.edu.vn/gpu150/pannet/predict', files={"file": (
    #     "filename", open(img_path, "rb"), "image/jpeg")}).json()

    words_list, prob_list = load_images_to_predict(detect_pannet, image)
    detect_vietocr = vietocr_json(words_list, prob_list)
    print(detect_vietocr)
    # output_txt(detect_pannet, detect_vietocr, output_txt_path)

    input_folder_img = 'input_img_test'
    visualize(image, img_path , detect_pannet, detect_vietocr)
    # result_extract_info_txt = extract_info(input_folder_img, OUTPUT_TXT, OUTPUT_PATH_POST, 'json_visualize_path')
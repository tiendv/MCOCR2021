# multithreaded.py
import math
import threading
import time
from queue import Queue
import requests
import json
import os
import cv2
# from visualize.visualize import visualize
from json_output import output_json_data_test
from utils.parser import get_config
from post_processing.extract_info import extract_info, add_info_compare
from visualize.output_json_train import output_json_for_train
from visualize.visualize_full_crop import *

# setup config
cfg = get_config()
cfg.merge_from_file('configs/service.yaml')

# create log_file, rcode
SUBMIT = cfg.SERVICE.SUBMIT
DATA_BTC_CSV = cfg.SERVICE.DATA_BTC_CSV
DETECT_TEXT_REG_TEXT = cfg.SERVICE.DETECT_TEXT_REG_TEXT
TASK1_URL = cfg.SERVICE.TASK1_URL
TASK2_URL = cfg.SERVICE.TASK2_URL
DETECT_RECEIPT_URL = cfg.SERVICE.DETECT_RECEIPT_URL
DETECT_RECEIPT_FASTER_RCNN = cfg.SERVICE.DETECT_RECEIPT_FASTER_RCNN
INPUT_IMAGES_NAME = cfg.SERVICE.INPUT_IMAGES_NAME
INPUT_FOLDER_IMAGES = os.path.join('raw_data_img',INPUT_IMAGES_NAME)
NUMBER_OF_THREADS = cfg.SERVICE.NUMBER_OF_THREADS
OUTPUT_PATH = cfg.SERVICE.OUTPUT_PATH
FOLDER_OUTPUT_TXT = os.path.join(OUTPUT_PATH,'detec_reg' ,INPUT_IMAGES_NAME)
FOLDER_OUTPUT_TXT_PROS = os.path.join(OUTPUT_PATH,'detec_reg_pros' ,INPUT_IMAGES_NAME)
OUTPUT_PATH_POST = os.path.join(cfg.SERVICE.OUTPUT_PATH_POST, INPUT_IMAGES_NAME)
if not os.path.exists(OUTPUT_PATH_POST):
    os.makedirs(OUTPUT_PATH_POST)
OUTPUT_PATH_PRE = cfg.SERVICE.OUTPUT_PATH_PRE
OUTPUT_PATH_VISUAL = os.path.join(cfg.SERVICE.OUTPUT_PATH_VISUAL,INPUT_IMAGES_NAME)
if not os.path.exists(OUTPUT_PATH_VISUAL):
    os.makedirs(OUTPUT_PATH_VISUAL)
OUTPUT_PATH_PRE_DETEC_RECEIPT = os.path.join(OUTPUT_PATH_PRE,INPUT_IMAGES_NAME+'/crop_detec_receipt')
if not os.path.exists(OUTPUT_PATH_PRE_DETEC_RECEIPT):
    os.makedirs(OUTPUT_PATH_PRE_DETEC_RECEIPT)
    print('OUTPUT_PATH_PRE_DETEC_RECEIPT', OUTPUT_PATH_PRE_DETEC_RECEIPT)


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
def output_txt_pros(data_task1, data_task2, output_path):
    bboxs = data_task1['result']
    words = data_task2['result']
    # print(words)
    result_txt_list = ''
    for box, word in zip(bboxs, words):
        result_txt_line = ''
        box = box['bbox']
        pros = word['pros']
        word = word['words']
        # print(box)
        # print(word)
        for b in box:
            result_txt_line += str(b) + ' '
            # print(result_txt_line)
        result_txt_line += str(pros) + ' ' + str(word)
        result_txt_list += result_txt_line + '\n'
    with open(output_path, 'w') as out:
        out.write(result_txt_list)
        print('output_path: ',output_path, '----ok----')

def make_request(url):
    img_path = url
    print(img_path)
    img_name = img_path.split('/')[-1]
    print(img_name)
    img = cv2.imread(img_path)

    if DETECT_RECEIPT_FASTER_RCNN:
        print('--------------CROP_IMG_DETECT_RECEIPT----------------')
        print(DETECT_RECEIPT_URL)
        detect_receipt = requests.post(DETECT_RECEIPT_URL, files={"file": (
            "filename", open(img_path, "rb"), "image/jpeg")}).json()
        receipt_box = detect_receipt['receipt']
        print('receipt_box', receipt_box)
        if receipt_box is not None:
            crop = img[receipt_box[1]:receipt_box[3], receipt_box[0]:receipt_box[2]]
            img_crop_request = cv2.imencode('.jpg', crop)[1]
            files = [
                ('img', img_crop_request.tostring())
            ]
            rotated_func = requests.request("POST", "http://service.aiclub.cs.uit.edu.vn/receipt/ript_rotate90/", files=files).text
            print('rotated_func', rotated_func)
            if rotated_func != 'None':
                dic_rotate_fuc = {'ROTATE_90_CLOCKWISE':cv2.ROTATE_90_CLOCKWISE, 'ROTATE_90_COUNTERCLOCKWISE':cv2.ROTATE_90_COUNTERCLOCKWISE, 'ROTATE_180':cv2.ROTATE_180}
                crop = cv2.rotate(crop, dic_rotate_fuc[rotated_func])
        img_out_path = os.path.join(OUTPUT_PATH_PRE_DETEC_RECEIPT, img_name)
        cv2.imwrite(img_out_path, crop)

        img_path = img_out_path
        img = cv2.imread(img_path)

    print('--------------DETECT_TEXT_PANNET----------------')
    detect_task1 = requests.post(TASK1_URL, files={"file": (
        "filename", open(img_path, "rb"), "image/jpeg")})
    print(detect_task1)
    detect_task1 = detect_task1.json()

    files = [
        ("file", ("filename", open(img_path, "rb"), "image/jpeg")),
        ('data', ('data', json.dumps(detect_task1), 'application/json')),
    ]

    print('--------------REG_TEXT_VIETOCR----------------')
    detect_task2 = requests.post(TASK2_URL, files=files).json()
    # print(detect_task2)
    output_txt_folder = FOLDER_OUTPUT_TXT
    if not os.path.exists(output_txt_folder):
        os.makedirs(output_txt_folder)
    txt_name = img_name.split('.')[0] + '.txt'
    print('txt_name',txt_name)
    output_txt_path = os.path.join(output_txt_folder, txt_name)
    print(output_txt_path)
    if not os.path.exists(FOLDER_OUTPUT_TXT_PROS):
        os.makedirs(FOLDER_OUTPUT_TXT_PROS)
    output_txt_pros_path = os.path.join(FOLDER_OUTPUT_TXT_PROS, txt_name)
    # output_folder_json = FOLDER_OUTPUT_JSON
    # if not os.path.exists(output_folder_json):
    #     os.mkdir(output_folder_json)
    # json_name = img_name.split('.')[0] + '.json'
    # print('json_name',json_name)
    # output_json_path = os.path.join(output_folder_json, json_name)
    # print(output_json_path)

    # cv2.imwrite(result_img_path, img)
    output_txt(detect_task1, detect_task2, output_txt_path)
    output_txt_pros(detect_task1, detect_task2, output_txt_pros_path)
    # visualize(img,img_path, detect_task1, detect_task2)
    # output_json_data_test(img, detect_task1, detect_task2, output_json_path, img_name)

def manage_queue():
    while True:
        current_url = url_queue.get()
        make_request(current_url)
        url_queue.task_done()


if __name__ == '__main__':
    if DETECT_TEXT_REG_TEXT:
        print('--------------detec and reg text--------------')
        number_of_threads = NUMBER_OF_THREADS

        print_lock = threading.Lock()

        url_queue = Queue()


        path, dirs, files = next(os.walk(INPUT_FOLDER_IMAGES))
        print(len(files))
        urls = []
        for fn in files:
            image_path = os.path.join(path, fn)
            urls.append(str(image_path))

        for i in range(number_of_threads):
            t = threading.Thread(target=manage_queue)
            t.daemon = True
            t.start()

        start = time.time()
        for current_url in urls:
            url_queue.put(current_url)
        url_queue.join()

        print("Execution time = {0:.5f}".format(time.time() - start))

    print('--------------Extract_info-----------------------')
    input_folder_img = OUTPUT_PATH_PRE_DETEC_RECEIPT if DETECT_RECEIPT_FASTER_RCNN else INPUT_FOLDER_IMAGES
    json_visualize_path = os.path.join(OUTPUT_PATH_VISUAL, 'json_visualize_for_test')
    if not os.path.exists(json_visualize_path):
        os.makedirs(json_visualize_path)
    output_add_info_path = os.path.join(OUTPUT_PATH_VISUAL, 'json_visualize_for_test_add_info')
    if not os.path.exists(output_add_info_path):
        os.makedirs(output_add_info_path)
    result_extract_info_txt = extract_info(input_folder_img, FOLDER_OUTPUT_TXT, OUTPUT_PATH_POST, json_visualize_path)
    # output_json_for_test('/home/huy/Downloads/' ,json_visualize_path)
    # input_txt_tac_gia_path = '/home/huy/Downloads/time_extract_val_v0.txt'
    # add_info_compare('cuong',json_visualize_path, input_txt_tac_gia_path, output_add_info_path)
    if not SUBMIT:
        json_visualize_train_path = os.path.join(OUTPUT_PATH_VISUAL, 'json_visualize_for_train')
        if not os.path.exists(json_visualize_train_path):
            os.makedirs(json_visualize_train_path)
        result_extract_info_txt = 'output_pipline/post_processing/train_1_16_10_14_rotated_receipt_text_khanh_075/results.txt'
        output_json_for_train(DATA_BTC_CSV, result_extract_info_txt, json_visualize_train_path)
        print('-----VISUALIZE-------')
        # visualize_for_train('raw_data_img/train_images', input_folder_img, FOLDER_OUTPUT_TXT,DATA_BTC_CSV , OUTPUT_PATH_VISUAL)
    else:
        img_visualize_test = os.path.join(OUTPUT_PATH_VISUAL, 'img_visualize_test')
        if not os.path.exists(img_visualize_test):
            os.makedirs(img_visualize_test)
        # print('-----VISUALIZE-------')
        # visualize(input_folder_img, FOLDER_OUTPUT_TXT, img_visualize_test)
        visualize_for_test(INPUT_FOLDER_IMAGES, INPUT_FOLDER_IMAGES, FOLDER_OUTPUT_TXT , img_visualize_test)
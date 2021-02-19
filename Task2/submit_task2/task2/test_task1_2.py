import requests
import json
import matplotlib as plt
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
# from visualize import visualize


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

# TASK1_URL = 'http://service.mmlab.uit.edu.vn/receipt/task1/predict'
TASK1_URL = 'http://service.aiclub.cs.uit.edu.vn/gpu150/pannet/predict'
TASK2_URL = 'http://service.mmlab.uit.edu.vn/receipt/task2/predict'
# TASK2_URL = 'http://service.aiclub.cs.uit.edu.vn/gpu150/vietocr/predict'
TASK1_URL = 'http://0.0.0.0:5005/predict'
# TASK2_URL = 'http://0.0.0.0:5006/predict'
# TASK1_URL = 'http://192.168.20.156:5012/predict'
OUTPUT = 'output_img'



if __name__ == "__main__":
    # 'mcocr_val_145115vxzfr.jpg'
    img_path = '/backup/mcocr_private_145120azxzl.jpg'
    img_name = img_path.split('/')[-1]
    print(img_path)

    img = cv2.imread(img_path)

    detect_task1 = requests.post(TASK1_URL, files={"file": (
        "filename", open(img_path, "rb"), "image/jpeg")}).json()
    print(detect_task1)


    files = [
        ("file", ("filename", open(img_path, "rb"), "image/jpeg")),
        ('data', ('data', json.dumps(detect_task1), 'application/json')),
    ]


    detect_task2 = requests.post(TASK2_URL, files=files).json()
    print(detect_task2)

    output_txt_file = os.path.join(OUTPUT, img_name.split('.')[0] + '.txt')

    # output_txt(detect_task1, detect_task2, output_txt_file)
    # visualize(img, img_path , detect_task1, detect_task2, )
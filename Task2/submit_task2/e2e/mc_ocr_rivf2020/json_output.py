import os
import cv2
import numpy as np
import json

def output_json_data_test(img, data_task1, data_task2, output_path, img_name):
    h, w, _ = img.shape
    bboxs = data_task1['result']
    words = data_task2['result']
    # print(words)
    result_txt_list = {"fname": img_name, "data":{"info":[], "box":[]}}
    box_lists = []
    for box, word in zip(bboxs, words):
        result_txt_line = ''
        box = box['bbox']
        word = word['words']
        if box[5] > box[1]:
            # print(word)
            # start_point = (int(box[0])-10, int(box[1])-20)
            # end_point = (int(box[4]), int(box[5]))
            bbox = [box[0], box[1], box[4], box[5]]
        else:
            # print(word)
            # start_point = (int(box[2])-10, int(box[3])-20)
            # end_point = (int(box[6]), int(box[7]))
            bbox = [box[2], box[3], box[6], box[7]]
        box_dicts = {"category_id":-1, "segmentation":box, "bbox":bbox,"area":-1 , "width":w, "height": h, "box_label": word}
        box_lists.append(box_dicts)
    result_txt_list["data"]["box"] = box_lists
    # print(result_txt_list)
    with open(output_path, 'w', encoding='utf8') as output_file:
        json.dump(result_txt_list, output_file, ensure_ascii=False)
        print('output_path: ',output_path, '----ok----')
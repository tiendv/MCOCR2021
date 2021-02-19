import json
import torch
from torch.autograd import Variable
import glob
import os
import csv
import cv2
import numpy as np
import argparse

from PIL import Image
import time

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

config = Cfg.load_config_from_name('vgg_seq2seq')

# config['weights'] = './transformerocr.pth'
# config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
config['device'] = 'cuda:0'
config['predictor']['beamsearch']=False

detector = Predictor(config)

def predict_this_box(config,img):
    s, pros = detector.predict(img, return_prob= True)
    # print(pros)
    return s, pros

def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def rotate_text(img, pts):
    height, width, _ = img.shape
    pts = pts.reshape(-1, 2)
    centroid = np.mean(pts, axis=0)

    left = pts[pts[:, 0] < centroid[0]]
    topleft = left[np.argmin(left, axis=0)[1]]
    botleft = left[np.argmax(left, axis=0)[1]]

    right = pts[pts[:, 0] > centroid[0]]
    topright = right[np.argmin(right, axis=0)[1]]
    botright = right[np.argmax(right, axis=0)[1]]

    w = int(np.linalg.norm(topright - topleft))
    h = int(np.linalg.norm(topright - botright))

    p1 = np.float32([topleft, topright, botright, botleft])
    p2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    M = cv2.getPerspectiveTransform(p1, p2)
    text = cv2.warpPerspective(img, M, (w, h))
    # cv2.imwrite('model/test/' + str(COUNT) + '.jpg', text)
    # COUNT = COUNT + 1
    return text

def isclose(a,b, th):
    return a == b or abs(a-b) < th 


def load_images_to_predict(data, image):
    COUNT = 0
    bboxs = data['result']
    words_list = []
    pros_list = []
    cnt = 0
    for box in bboxs:
        box = box['bbox']
        box_backup = box[:]
        # print('box:',box)
        box_them = 6
        box_giam = 4
        #code xoay text cua khanh   
        # pts = np.array([[[box[0],box[1]],[box[2],box[3]],[box[4],box[5]],[box[6],box[7]]]])
        # text = rotate_text(image, pts)
        # boxImg = Image.fromarray(text)

        # code chưa xoay 
        try:
            img = image[box[1]-box_giam:box[5]+box_them,box[0]-box_giam:box[4]+box_them]
            boxImg = Image.fromarray(img)
            words_test, pros = predict_this_box(config, boxImg)
            # if pros < 0.55 or box[5] > box[3] and box[7] > box[1]:
            #     pts = np.array([[[box_backup[0],box_backup[1]],[box_backup[2],box_backup[3]],[box_backup[4],box_backup[5]],[box_backup[6],box_backup[7]]]])
            #     text = rotate_text(image, pts)
            #     boxImg = Image.fromarray(text)
            #     words_test, pros = predict_this_box(config, boxImg)
            #     if pros > 0.55:
            #         words_list.append(words_test)
            #         pros_list.append(pros)
            #         print('fix bug box nghien ve ben phai', words_test, box_backup, pros)
            #         continue
            if not isclose(box[5],box[7], 15):
                box[1] = box[1] -box_giam
                box[5] = box[5] +box_them
                box[0] = box[0] - box_giam
                box[4] = box[4] + box_them
                box[2] = box[2] + box_them
                box[3] = box[3] - box_giam
                box[6] = box[6] - box_giam
                box[7] = box[7] + box_them

                
                pts = np.array([[[box[0],box[1]],[box[2],box[3]],[box[4],box[5]],[box[6],box[7]]]])
                text = rotate_text(image, pts)
                boxImg = Image.fromarray(text)
                words_test, pros = predict_this_box(config, boxImg)
                if '0000000000' in words_test or pros < 0.55:
                    boxImg = Image.fromarray(img)
        except:
            img = image[box[3]-box_giam:box[7]+box_them,box[2]-box_giam:box[6]+box_them]
            # boxImg = Image.fromarray(img)
            
            try:
                boxImg = Image.fromarray(img)
                if not isclose(box[1],box[7], 15):

                    box[1] = box[1] + box_them
                    box[5] = box[5] + box_them
                    box[0] = box[0] - box_giam
                    box[4] = box[4] - box_giam
                    box[2] = box[2] - box_giam
                    box[3] = box[3] - box_giam
                    box[6] = box[6] + box_them
                    box[7] = box[7] + box_them
                    
                    pts = np.array([[[box[0],box[1]],[box[2],box[3]],[box[4],box[5]],[box[6],box[7]]]])
                    text = rotate_text(image, pts)
                    boxImg = Image.fromarray(text)
                    words_test, pros = predict_this_box(config, boxImg)
                    if '0000000000' in words_test or pros < 0.55:
                        boxImg = Image.fromarray(img)
            except:
                pts = np.array([[[box[0],box[1]],[box[2],box[3]],[box[4],box[5]],[box[6],box[7]]]])
                text = rotate_text(image, pts)
                boxImg = Image.fromarray(text)
                
        # try:
        boxImg.save('task2/model/test/' + str(COUNT) + '.jpg')
        # width, height = boxImg.size
        # boxImg = boxImg.resize((width*1.5,height*1.5))
        # boxImg.save('model/test/' + str(COUNT) + '_scale.jpg')
        # color = boxImg.getpixel((3,3))
        # print(color)
        # boxImg = add_margin(boxImg, 0,20,0,20, color)

        # boxImg.save('model/test/' + str(COUNT) + '_padding.jpg')
        COUNT = COUNT + 1
        words,pros  = predict_this_box(config, boxImg)
        if pros < 0.55:
            words = 'erorr_bi_mo'
        words_list.append(words)
        pros_list.append(pros)
        # print(box, words, pros)
        # except:
            # words_list.append('error')
            # continue
    return words_list, pros_list
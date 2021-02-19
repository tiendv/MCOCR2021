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
# config['trainer']['checkpoint'] = '/dataset/Students/thuyentd/VietOcr/vgg_seq2seq_receipt_31122020checkpoint.pth'

detector = Predictor(config)

def predict_this_box(config,img):
    s, pros = detector.predict(img, return_prob= True)
    return s, pros

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

    return text

def load_images_to_predict_backup(data, image):
    bboxs = data['result']
    words_list = []
    cnt = 0
    for box in bboxs:
        box = box['bbox']
        print(box)
        
        #code xoay text cua khanh   
        # pts = np.array([[[box[0],box[1]],[box[2],box[3]],[box[4],box[5]],[box[6],box[7]]]])
        # text = rotate_text(image, pts)
        # boxImg = Image.fromarray(text)

        # code chưa xoay 
        try:
            img = image[box[1]-3:box[5]+5,box[0]-3:box[4]+5]
            boxImg = Image.fromarray(img)
        except:
            img = image[box[3]-3:box[7]+5,box[2]-3:box[6]+5]
            try:
                boxImg = Image.fromarray(img)
            except:
                # pts = np.array([[[box[0],box[1]],[box[2],box[3]],[box[4],box[5]],[box[6],box[7]]]])
                # text = rotate_text(image, pts)
                # boxImg = Image.fromarray(text)
                print('error')
                
        try:
            words = predict_this_box(config, boxImg)
            words_list.append(words)
        except:
            words_list.append('')
            continue
    return words_list

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
        # print(box)
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
                words_test = predict_this_box(config, boxImg)
                if '0000000000' in words_test:
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
                    words_test = predict_this_box(config, boxImg)
                    if '0000000000' in words_test:
                        boxImg = Image.fromarray(img)
            except Exception as e:
                # boxImg = Image.fromarray(img)
                # if not isclose(box[1],box[7], 7):
                print(e)
                # box[1] = box[1] + 5
                # box[5] = box[5] + 5
                # box[0] = box[0] - 3
                # box[4] = box[4] - 3
                # box[2] = box[2] - 3
                # box[3] = box[3] - 3
                # box[6] = box[6] + 5
                # box[7] = box[7] + 5
                pts = np.array([[[box[0],box[1]],[box[2],box[3]],[box[4],box[5]],[box[6],box[7]]]])
                text = rotate_text(image, pts)
                boxImg = Image.fromarray(text)
                
        # try:
        # boxImg.save('model/test/' + str(COUNT) + '.jpg')
        # COUNT = COUNT + 1
        # width, height = boxImg.size
        # boxImg = boxImg.resize((width*2,height*2))
        words, pros = predict_this_box(config, boxImg)
        words_list.append(words)
        pros_list.append(pros)
        if pros < 0.55:
            words = 'None Content'
        print(box, words, pros)
        # except:
            # words_list.append('error')
            # continue
    print(pros_list)
    return words_list, pros_list
# -*- coding: utf-8 -*-
# @Time    : 2019/8/24 12:06
# @Author  : zhoujun

import torch
from torchvision import transforms
import os
import cv2
import time
import json
import requests

from models import get_model

from post_processing import decode

def decode_clip(preds, scale=1, threshold=0.7311, min_area=5):
    import pyclipper
    import numpy as np
    preds[:2, :, :] = torch.sigmoid(preds[:2, :, :])
    preds = preds.detach().cpu().numpy()
    text = preds[0] > threshold  # text
    kernel = (preds[1] > threshold) * text  # kernel

    label_num, label = cv2.connectedComponents(kernel.astype(np.uint8), connectivity=4)
    bbox_list = []
    for label_idx in range(1, label_num):
        points = np.array(np.where(label == label_idx)).transpose((1, 0))[:, ::-1]
        if points.shape[0] < min_area:
            continue
        rect = cv2.minAreaRect(points)
        poly = cv2.boxPoints(rect).astype(int)

        d_i = cv2.contourArea(poly) * 1.5 / cv2.arcLength(poly, True)
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        shrinked_poly = np.array(pco.Execute(d_i))
        if shrinked_poly.size == 0:
            continue
        rect = cv2.minAreaRect(shrinked_poly)
        shrinked_poly = cv2.boxPoints(rect).astype(int)
        if cv2.contourArea(shrinked_poly) < 800 / (scale * scale):
            continue

        bbox_list.append([shrinked_poly[1], shrinked_poly[2], shrinked_poly[3], shrinked_poly[0]])
    return label, np.array(bbox_list)


class Pytorch_model:
    def __init__(self, model_path, gpu_id=None):
        '''
        初始化pytorch模型
        :param model_path: 模型地址(可以是模型的参数或者参数和计算图一起保存的文件)
        :param gpu_id: 在哪一块gpu上运行
        '''
        self.gpu_id = gpu_id

        if self.gpu_id is not None and isinstance(self.gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:%s" % self.gpu_id)
        else:
            self.device = torch.device("cpu")
        print('device:', self.device)
        checkpoint = torch.load(model_path, map_location=self.device)

        config = checkpoint['config']
        config['arch']['args']['pretrained'] = False
        self.net = get_model(config)

        self.img_channel = config['data_loader']['args']['dataset']['img_channel']
        self.net.load_state_dict(checkpoint['state_dict'])
        self.net.to(self.device)
        self.net.eval()

    def predict(self, img, short_size: int = 736):
        '''
        对传入的图像进行预测，支持图像地址,opecv 读取图片，偏慢
        :param img: 图像地址
        :param is_numpy:
        :return:
        '''
        # assert os.path.exists(img), 'file is not exists'
        # img = cv2.imread(img)
        if self.img_channel == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        scale = short_size / min(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)
        # 将图片由(w,h)变为(1,img_channel,h,w)
        tensor = transforms.ToTensor()(img)
        tensor = tensor.unsqueeze_(0)

        tensor = tensor.to(self.device)
        with torch.no_grad():
            if str(self.device).__contains__('cuda'):
                torch.cuda.synchronize(self.device)
            start = time.time()
            preds = self.net(tensor)[0]
            if str(self.device).__contains__('cuda'):
                torch.cuda.synchronize(self.device)
            preds, boxes_list = decode(preds)
            scale = (preds.shape[1] / w, preds.shape[0] / h)
            if len(boxes_list):
                boxes_list = boxes_list / scale
            t = time.time() - start
        return preds, boxes_list, t

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

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # from utils.util import show_img, draw_bbox

    os.environ['CUDA_VISIBLE_DEVICES'] = str('0')

    model_path = 'PANNet_model_pretrain_SOIRE.pth'

    
    img_path = 'mcocr_private_145120azxzl.jpg'
    image = cv2.imread(img_path)
    
    model = Pytorch_model(model_path, gpu_id=0)
    preds, boxes_list, t = model.predict(image)
    print(boxes_list)
    

    detect_pannet = pannet_json(boxes_list)
    print(detect_pannet)
    
    TASK2_URL = 'http://service.mmlab.uit.edu.vn/receipt/task2/predict'
    files = [
        ("file", ("filename", open(img_path, "rb"), "image/jpeg")),
        ('data', ('data', json.dumps(detect_pannet), 'application/json')),
    ]


    detect_task2 = requests.post(TASK2_URL, files=files).json()
    print(detect_task2)

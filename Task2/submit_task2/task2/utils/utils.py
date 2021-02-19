import cv2 
import os 
import requests
import time
import random
import numpy
import sys
from time import gmtime, strftime

def load_class_names(filename):
    with open(filename, 'r', encoding='utf8') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

def get_frame(video_file, URL):
    camera=cv2.VideoCapture(video_file)

    while True:
        retval, im = camera.read()

        # gen name
        my_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
        number = str(random.randint(0, 10000))
        img_name = my_time + '_' + number + '.jpg'
        img_path = os.path.join('backup', img_name)
        cv2.imwrite(img_path, im)

        response = requests.post(URL, files={"file": (img_name, open(img_path, "rb"), "image/jpeg")}).json()

        image_path = response['visual_path']
        image = cv2.imread(image_path)

        imgencode=cv2.imencode('.jpg',image)[1]
        
        stringData=imgencode.tostring()

        yield (b'--frame\r\n'
            b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')

    del(camera)

def get_image(image_path):
    # image_path = os.path.join("backup", filename)
    # image = cv2.imread(image_path)
    while True:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (480, 270))
        imgencode=cv2.imencode('.jpg',image)[1]
        stringData=imgencode.tostring()

        yield (b'--frame\r\n'
            b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')
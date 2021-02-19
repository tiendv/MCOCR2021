import os
import lmdb  # install lmdb by "pip install lmdb"
import cv2
import numpy as np
import glob
import json
from PIL import Image, ImageDraw, ImageFont

def visualize(img, img_path , data_task1, data_task2):
    bboxs = data_task1['result']
    words = data_task2['result']
    # img = cv2.imread(img_path)

    img_1 = np.ones([img.shape[0],img.shape[1],3],dtype=np.uint8)*255
    fontpath = "unicode.publish.UVNVietSach_R.TTF" 
    font = ImageFont.truetype(fontpath, 13)
    im = Image.open(img_path) # img1 anh task 1
    draw1 = ImageDraw.Draw(im)

    img_pil = Image.fromarray(img_1)
    draw = ImageDraw.Draw(img_pil)
    b,g,r,a = 0,0,255,0
    name_img = img_path.split('/')[-1]
    for box, word in zip(bboxs, words):
        box = box['bbox']
        word = word['words']
        if box[5] > box[1]:
            # print(word)
            start_point = (int(box[0])-10, int(box[1])-20)
            end_point = (int(box[4]), int(box[5]))
        else:
            # print(word)
            start_point = (int(box[2])-10, int(box[3])-20)
            end_point = (int(box[6]), int(box[7]))
        draw.text(start_point,  word, font = font, fill = (b, g, r, a))
        draw1.rectangle([start_point,end_point], outline ="red") 
        img_2 = np.array(img_pil) # img2 anh task 2
    # cv2.imwrite('task2_'+str(name_img),img_2)
    new_image = Image.new('RGB',(2*im.size[0], im.size[1]), (250,250,250))
    new_image.paste(im,(0,0))
    new_image.paste(img_pil,(im.size[0],0))
    output = 'visualize/'
    if not os.path.exists(output):
        os.mkdir(output)
    new_image.save(output + name_img,"JPEG")
    return
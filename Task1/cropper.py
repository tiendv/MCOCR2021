import requests
import json
import numpy as np
import cv2
import os
from tqdm import tqdm

def crop_receipt(raw_img):
    """Crop receipt from a raw image captured by phone

    Args:
        raw_img ([np.array]): Raw image containing receipt

    Returns:
        cropped_receipt ([np.array]): The image of cropped receipt 
    """    

    CROP_RECEIPT_URL = 'http://service.aiclub.cs.uit.edu.vn/receipt/ript_detect'
    ROTATE_RECEIPT_URL = 'http://service.aiclub.cs.uit.edu.vn/receipt/ript_rotate90/'

    _, img_encoded = cv2.imencode('.jpg', raw_img)

    detect_receipt = requests.post(CROP_RECEIPT_URL, files={"file": (
        "filename", img_encoded.tostring(), "image/jpeg")}).json()
    receipt_box = detect_receipt['receipt']
    if receipt_box is not None:
        crop = raw_img[receipt_box[1]:receipt_box[3], receipt_box[0]:receipt_box[2]]
        img_crop_request = cv2.imencode('.jpg', crop)[1]
        files = [
            ('img', img_crop_request.tostring())
        ]
        rotated_func = requests.request("POST", "http://service.aiclub.cs.uit.edu.vn/receipt/ript_rotate90/", files=files).text
        rotated_func = rotated_func.split('\n')
        if rotated_func[0] != 'None' and float(rotated_func[1]) > 0.6:
            dic_rotate_fuc = {'ROTATE_90_CLOCKWISE':cv2.ROTATE_90_CLOCKWISE, 'ROTATE_90_COUNTERCLOCKWISE':cv2.ROTATE_90_COUNTERCLOCKWISE, 'ROTATE_180':cv2.ROTATE_180}
            crop = cv2.rotate(crop, dic_rotate_fuc[rotated_func[0]])
        return crop
    
    return raw_img
import requests
import json
import numpy as np
import cv2

img_path = '/home/huy/TEST_SUBMIT/end_to_end_submit/raw_data_img/test/mcocr_val_145114ixmyt.jpg'

img = cv2.imread(img_path)
DETECT_RECEIPT_URL = 'http://service.aiclub.cs.uit.edu.vn/receipt/ript_detect'
print('--------------CROP_IMG_DETECT_RECEIPT----------------')
print(DETECT_RECEIPT_URL)
detect_receipt = requests.post(DETECT_RECEIPT_URL, files={"file": (
    "filename", open(img_path, "rb"), "image/jpeg")}).json()
receipt_box = detect_receipt['receipt']
print('receipt_box', receipt_box)
img_out_path = os.path.join(OUTPUT_PATH_PRE_DETEC_RECEIPT, img_name)
if receipt_box is not None:
    crop = img[receipt_box[1]:receipt_box[3], receipt_box[0]:receipt_box[2]]
    img_crop_request = cv2.imencode('.jpg', crop)[1]
    files = [
        ('img', img_crop_request.tostring())
    ]
    rotated_func = requests.request("POST", "http://service.aiclub.cs.uit.edu.vn/receipt/ript_rotate90/", files=files).text
    print('rotated_func', rotated_func)
    rotated_func = rotated_func.split('\n')
    if rotated_func[0] != 'None' and float(rotated_func[1]) > 0.6:
        dic_rotate_fuc = {'ROTATE_90_CLOCKWISE':cv2.ROTATE_90_CLOCKWISE, 'ROTATE_90_COUNTERCLOCKWISE':cv2.ROTATE_90_COUNTERCLOCKWISE, 'ROTATE_180':cv2.ROTATE_180}
        crop = cv2.rotate(crop, dic_rotate_fuc[rotated_func[0]])
    cv2.imwrite(img_out_path, crop)
else:
    cv2.imwrite(img_out_path, img)
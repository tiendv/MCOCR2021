import cv2
import numpy as np 
import time
import logging
import traceback
import os
import io
import requests
import random
import json
from time import gmtime, strftime

from flask import Flask, render_template, Response, request, jsonify

from utils.parser import get_config
# from utils.utils import load_class_names, get_image

import cv2
import numpy as np 

from model.predict import load_images_to_predict

# create backup dir
if not os.path.exists('backup'):
    os.mkdir('backup')

# create json dir
if not os.path.exists('json_dir'):
    os.mkdir('json_dir')

# setup config
cfg = get_config()
cfg.merge_from_file('configs/service.yaml')

# create log_file, rcode
TASK1_URL = cfg.SERVICE.TASK1_URL
TASK2_URL = cfg.SERVICE.TASK2_URL
TASK3_URL = cfg.SERVICE.TASK3_URL
LOG_PATH = cfg.SERVICE.LOG_PATH
BACKUP = cfg.SERVICE.BACKUP_DIR
HOST = cfg.SERVICE.SERVICE_IP
PORT = cfg.SERVICE.SERVICE_PORT

if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_PATH)
logging.basicConfig(filename=os.path.join(LOG_PATH, str(time.time())+".log"), filemode="w", level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console = logging.StreamHandler()
console.setLevel(logging.ERROR)
logging.getLogger("").addHandler(console)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        data = json.load(request.files['data'])
        image_file = file.read()
        img = cv2.imdecode(np.frombuffer(image_file, dtype=np.uint8), -1)

        words_list = load_images_to_predict(data, img)

        result = []
        cnt = 0 
        
        while cnt < len(words_list):
            words = words_list[cnt]
            my_dict = {
                "words": words
            }
            result.append(my_dict)
            cnt += 1
    with open(BACKUP+'/result.json', 'w') as outfile:
        json.dump(result, outfile)
    return jsonify(result = result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5010, debug=True)

from flask import Flask, render_template, request, jsonify

import cv2
import numpy as np 
from predict import Pytorch_model
import predict

app = Flask(__name__)

model = Pytorch_model("PANNet_model_pretrain_SOIRE.pth", gpu_id=0)

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        image_file = file.read()
        image = cv2.imdecode(np.frombuffer(image_file, dtype=np.uint8), -1)

        preds, boxes_list, t = model.predict(image)

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

        # my_color = model.predict(x, batch_size=1)
        # my_color = color_labels[np.argmax(my_color)]
        # result = str(result)
        print(result)
        return jsonify(result = result)
        #return result
        # return 'OK'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5010, debug=False)

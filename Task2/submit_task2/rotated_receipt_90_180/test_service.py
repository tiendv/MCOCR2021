import requests
import cv2

img = cv2.imread('20201116_164641.jpg')

img = cv2.imencode('.jpg', img)[1]

files = [
    ('img', img.tostring())
]
# http://0.0.0.0:6493/
# http://service.aiclub.cs.uit.edu.vn/receipt/ript_rotate90/

rotated_func = requests.request("POST", "http://0.0.0.0:6400/rotate-img", files=files).text
print('rotated_func', rotated_func) 
import requests
import cv2

url = "http://service.aiclub.cs.uit.edu.vn/receipt/ript_rotate90/"

img_path = '/home/huy/TEST_SUBMIT/end_to_end_submit/output_pipline/pre_processing/full_dataset_my_v2_fixed/crop_detec_receipt/uit_data_024.jpg'
    
def request_img_path():
    payload = {
        'img_path': img_path,
    }
    files = []

    return requests.request("POST", url, data=payload, files=files)
    
    
def request_img():
    img = cv2.imread(img_path)
    print(img)
    img = cv2.imencode('.jpg', img)[1]

    files = [
        ('img', img.tostring())
    ]
    return requests.request("POST", url, files=files)


if __name__ == '__main__':
    response = request_img()

    if response.status_code == 200:
        print(response.text)
        p = response.text.split('\n')[0]
        print(p)
    else:
        print('ERROR: ' + response.text)

# import requests
# import cv2

# url = "http://192.168.20.151:6493/"

# img_path = '/home/huy/TEST_SUBMIT/end_to_end_submit/raw_data_img/output_deskew_test_25_15_04/mcocr_val_145114anqqj.jpg'
    
# def request_img_path():
#     payload = {
#         'img_path': img_path,
#     }
#     files = []

#     return requests.request("POST", url + 'rotate-img', data=payload, files=files)
    
    
# def request_img():
#     img = cv2.imread(img_path)
#     img = cv2.imencode('.jpg', img)[1]

#     files = [
#         ('img', img.tostring())
#     ]
#     return requests.request("POST", url + "rotate-img", files=files)


# if __name__ == '__main__':
#     response = request_img_path()

#     if response.status_code == 200:
#         print(response.text)
#     else:
#         print('ERROR: ' + response.text)
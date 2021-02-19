import cv2
import numpy as np
import time
import os
from multiprocessing import Pool
# from google.colab.patches import cv2_imshow
from sklearn import linear_model

import random
random.seed(42)

def load_bbox(label_path):
    lines = open(label_path, 'r').read().splitlines()
    lines = [line.split()[:8] for line in lines]
    lines = [list(map(int, line)) for line in lines]
    lines = [list(zip(line[::2], line[1::2])) for line in lines]
    lines = np.array(lines)
    return lines

def arrange_bbox(pts):
    pts = pts.reshape(-1, 4, 2)
    n = np.arange(pts.shape[0])
    centroids = np.mean(pts, axis=1)

    left_mask = pts[:, :, 0] < centroids[:, 0, np.newaxis]
    assert all(np.sum(left_mask, axis=1) == 2), "Xuat hien bbox nghieng hon 45 do :(\nTO-DO: fix this case"
    left = pts[left_mask].reshape(-1, 2, 2)
    topleft = left[n, np.argmin(left, axis=1)[:, 1]]
    botleft = left[n, np.argmax(left, axis=1)[:, 1]]

    right_mask = pts[:, :, 0] > centroids[:, 0, np.newaxis]
    assert all(np.sum(right_mask, axis=1) == 2), "Xuat hien bbox nghieng hon 45 do :(\nTO-DO: fix this case"
    right = pts[right_mask].reshape(-1, 2, 2)
    topright = right[n, np.argmin(right, axis=1)[:, 1]]
    botright = right[n, np.argmax(right, axis=1)[:, 1]]

    pts = np.array([topleft, topright, botright, botleft]).transpose(1, 0, 2)
    return pts, centroids

def rotate_text(pts, centroids):
    AB = pts[:, 1] - pts[:, 0]
    AD = pts[:, -1] - pts[:, 0]
    w = np.linalg.norm(AB, axis=1).copy()
    h = np.linalg.norm(AD, axis=1).copy()
    ar = w/h

    # score = độ tin cậy của 1 bbox, càng "dài" và "to" thì càng đáng tin
    score = ar/np.amax(ar) + 3 * h/np.amax(h)

    cos = AB[:, 0] / w
    sin = np.sqrt(1 - np.square(cos))
    sin[AB[:, 1] < 0] *= -1
    # Tìm góc xoay của bbox đáng tin cậy nhất
    average_sin = np.average(sin, weights=score)
    # average_sin = sin[np.argmax(score)]
    average_cos = np.sqrt(1 - np.square(average_sin))

    # v = vector đơn vị xoay trung bình = [cos(x), sin(x)]
    v = np.array([average_cos, average_sin])
    # p = vector pháp tuyến của v = [-sin(x), cos(x)]
    p = np.array([-average_sin, average_cos])

    return w, h, v, p, average_cos, average_sin

def crop_text(img, pts, rotate_only=True, SCALE_FACTOR = 1.25):
    pts, centroids = arrange_bbox(pts)
    w, h, v, p, cos, sin = rotate_text(pts, centroids)

    h = np.broadcast_to(h[:, np.newaxis], (*h.shape, 2)).copy()
    w = np.broadcast_to(w[:, np.newaxis], (*w.shape, 2)).copy()
    x = w*v
    y = h*p

    homo_src = np.float32([
        [+1, +1],
        [+1, -1],
        [-1, -1],
        [-1, +1],
    ])
    homo_src += np.float32([img.shape[1::-1]])/2
    homo_dst = np.float32([
        [+ cos + sin, - sin + cos],
        [+ cos - sin, - sin - cos],
        [- cos - sin, + sin - cos],
        [- cos + sin, + sin + cos],
    ])
    homo_dst += np.float32([img.shape[1::-1]])/2
    M = cv2.getPerspectiveTransform(homo_src, homo_dst)
    rotated_img = cv2.warpPerspective(img, M, img.shape[1::-1])

    if rotate_only: return rotated_img

    pts[:, 0] = centroids + (pts[:, 0] - centroids) * np.array([1, SCALE_FACTOR])
    pts[:, 1] = centroids + (pts[:, 1] - centroids) * np.array([1, SCALE_FACTOR])
    pts[:, 2] = centroids + (pts[:, 2] - centroids) * np.array([1, SCALE_FACTOR**1.5])
    pts[:, 3] = centroids + (pts[:, 3] - centroids) * np.array([1, SCALE_FACTOR**1.5])
    h *= (SCALE_FACTOR + SCALE_FACTOR**1.5) / 2

    height, width = img.shape[:2]

    img_bbox = cv2.polylines(img.copy(), pts, True, (0, 0, 255), thickness=1)
    # cv2_imshow(img_bbox)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 4))

    text = []
    debug_text = []
    for p, _w, _h in zip(pts, w[:, 0], h[:, 0]):
        p1 = p.astype(np.float32)
        p2 = np.float32([
            np.array([0, 0]),
            np.array([_w, 0]),
            np.array([_w, _h]),
            np.array([0, _h]),
        ])
        _w, _h = int(_w), int(_h)
        M = cv2.getPerspectiveTransform(p1, p2)
        cropped = cv2.warpPerspective(img, M, (_w, _h))
        t = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        # t = cv2.medianBlur(t, 5)
        t = cv2.GaussianBlur(t, (17, 17), 0)
        t = cv2.adaptiveThreshold(t, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 5)

        t = cv2.erode(t, kernel)
        # t = cv2.dilate(t, kernel)

        t = 1 - np.clip(t, 0, 1)
        nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(t, connectivity=4)


        t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
        t = np.zeros_like(t, dtype=np.uint8)

        
        # left = stats[:, cv2.CC_STAT_LEFT]
        # right = stats[:, cv2.CC_STAT_LEFT] + stats[:, cv2.CC_STAT_WIDTH]
        # top = stats[:, cv2.CC_STAT_TOP]
        # bottom = stats[:, cv2.CC_STAT_TOP] + stats[:, cv2.CC_STAT_HEIGHT]
        # height = stats[:, cv2.CC_STAT_HEIGHT]

        potential_mask = np.abs(_h/2 - centroids[:, 1])/_h < 0.25
        potential_mask = np.where(potential_mask)[0]

        # dist = np.min([
        #     np.abs(left - _w//2),
        #     np.abs(right - _w//2),
        #     np.abs(top - _h//2),
        #     np.abs(bottom - _h//2),
        # ], axis=0).astype(np.float32)
        # dist[np.logical_and(np.logical_and(left <= _w//2, _w//2 <= right), np.logical_and(top <= _h//2, _h//2 <= bottom))] = 0

        # dist -= height * np.amax(dist) / np.amax(height)
        # dist[0] = np.inf

        # min_dist = np.argmin(dist)

        filter = np.zeros_like(t)
        segmap = cv2.cvtColor(filter, cv2.COLOR_BGR2GRAY)
        seg_min = _w + 1
        seg_max = 0
        homo_list = []
        lines = []

        for i, (stat, centroid) in enumerate(zip(stats, centroids)):
            if i==0: continue

            if i in potential_mask:
                color = (0, 255, 255)
                filter[labels==i] = (0, 0, 255)
                segmap[labels==i] = 1
                seg_min = min(seg_min, stat[0])
                seg_max = max(seg_max, stat[0] + stat[2])
            else:
                color = int(random.random()*255), int((0.5 + random.random()/2)*255), int((0.4 + random.random()/5)*255)
            t[labels==i] = color
            t = cv2.circle(t, tuple(centroid.astype(int)), 4, ((color[0]+90) % 180, color[1], color[2]), thickness=-1)

        segmap = np.array(np.where(segmap > 0)).transpose()

        # Tìm đường cong của box
        X, unique_index = np.unique(segmap[:, 1], return_index=True)
        X = X.reshape(-1, 1)
        X_0 = np.ones_like(X)
        X_1 = X
        X_2 = np.square(X)
        X_3 = X_2 * X
        X_bar = np.hstack((X_0, X_1, X_2, X_3))
        y = segmap[unique_index, 0]

        regr = linear_model.LinearRegression(fit_intercept=False)
        regr.fit(X_bar, y)

        # Tịnh tiến đường ở trên
        y_hat = X_bar @ regr.coef_.reshape(4, 1)
        top_loss = y - y_hat.reshape(-1)
        top_line = regr.coef_.copy()
        top_line[0] += np.amin(top_loss)

        # Tịnh tiến đường ở dưới
        X, unique_index = np.unique(segmap[::-1, 1], return_index=True)
        X = X.reshape(-1, 1)
        X_0 = np.ones_like(X)
        X_1 = X
        X_2 = np.square(X)
        X_3 = X_2 * X
        X_bar = np.hstack((X_0, X_1, X_2, X_3))
        y = segmap[len(segmap) - 1 - unique_index, 0]
        y_hat = X_bar @ regr.coef_.reshape(4, 1)
        bot_loss = y - y_hat.reshape(-1)
        bot_line = regr.coef_.copy()
        bot_line[0] += np.amax(bot_loss)

        curve_height = bot_line[0] - top_line[0]

        # Visualize đường cong
        X_1 = np.linspace(seg_min, seg_max, num=10).reshape(-1, 1)
        X_0 = np.ones_like(X_1)
        X_2 = np.square(X_1)
        X_3 = X_1 ** 3
        X_bar = np.hstack((X_0, X_1, X_2, X_3))
        y = X_bar @ top_line.reshape(4, 1)
        line = np.hstack((X_1, y))
        lines.append(line.astype(np.int32))
        y = X_bar @ bot_line.reshape(4, 1)
        line = np.hstack((X_1, y))
        lines.append(line.astype(np.int32))

        y = X_bar @ regr.coef_.reshape(4, 1)
        line = np.hstack((X_1, y))
        lines.append(line.astype(np.int32))
        filter = cv2.polylines(filter, lines, False, (0, 255, 0), thickness=2)

        curve_list = []

        X_1 = np.linspace(seg_min, seg_max, num=10).reshape(-1, 1)
        for xmin, xmax in zip(X_1, X_1[1:]):
            x_1 = np.array([xmin, xmax])
            x_0 = np.ones_like(x_1)
            x_2 = np.square(x_1)
            x_3 = x_1 ** 3
            x_bar = np.hstack((x_0, x_1, x_2, x_3))
            y_top = x_bar @ top_line.reshape(4, 1)
            y_bot = x_bar @ bot_line.reshape(4, 1)
            homo_src = np.float32([
                [xmin, y_top[0,0]],
                [xmax, y_top[1,0]],
                [xmax, y_bot[1,0]],
                [xmin, y_bot[0,0]],
            ])
            homo_dst = np.float32([
                [0, 0],
                [xmax-xmin, 0],
                [xmax-xmin, curve_height],
                [0, curve_height],
            ])
            M = cv2.getPerspectiveTransform(homo_src, homo_dst)
            curve = cv2.warpPerspective(cropped, M, (int(xmax-xmin), int(curve_height)))
            curve_list.append(curve)

        final_curve = np.hstack(tuple(curve_list))
        final_curve = cv2.resize(final_curve, (final_curve.shape[1] * _h // final_curve.shape[0], _h))

        t = cv2.cvtColor(t, cv2.COLOR_HSV2BGR)
        t = cv2.circle(t, (_w//2, _h//2), 4, (0, 255, 0), thickness=-1)
        
        separator = np.full((t.shape[0], 10, 3), 255, dtype=np.uint8)
        t = np.hstack((cropped, separator, t, separator, filter, separator, final_curve))

        text.append(final_curve)
        debug_text.append(t)



    return rotated_img, text, debug_text


if __name__ == '__main__':
    base_img = 'output_pipline/pre_processing/mcocr_private_test_data/crop_detec_receipt'
    # base_rec = '/content/drive/MyDrive/Contest Folder/MC-OCR_RIVF2020/WorkingSpace/Cropping/result_detec_receipt_crop/txt_result'
    base_text = 'output_pipline/detec_reg/mcocr_private_test_data'
    base_target_img = "raw_data_img/mcocr_private_test_data_detec_receipt"
    # img = cv2.imread('/mcocr_warmup_010cefb7d7b0f5a62e7659cf7685fd57_00317.jpg')
    if not os.path.exists(base_target_img):
        os.mkdir(base_target_img)

    path, dirs, files = next(os.walk(base_img))
    print(len(files))
    urls = []
    for fn in files:
        try:
            img_name = fn.split('.')[0]
            img = cv2.imread(base_img +'/{}.jpg'.format(img_name))
            pts = load_bbox( base_text +'/{}.txt'.format(img_name))

            img_bbox = cv2.polylines(img.copy(), pts, True, (0, 0, 255), thickness=2)
            img_bbox = cv2.resize(img_bbox, (img_bbox.shape[1] * 416 // img_bbox.shape[0], 416))

        
            rotated_img = crop_text(img, pts)
            # print(time.time() - t)

            # rotated_img: hình sau khi xoay
            # text: list chứa các text sau khi crop và duỗi thẳng

            # rotated_img = cv2.resize(rotated_img, (rotated_img.shape[1] * 416 // rotated_img.shape[0], 416))

            print(os.path.join(base_target_img,fn))
            cv2.imwrite(os.path.join(base_target_img,fn), rotated_img)
        except:
            cv2.imwrite(os.path.join(base_target_img,fn), img)
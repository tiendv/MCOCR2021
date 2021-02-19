import cv2
import numpy as np 


def draw_bbox(image, list_bbox, list_scores, list_class_ids):
    i = 0
    len_list_bbox = len(list_bbox)
    while i < len_list_bbox:
        bbox = list_bbox[i]
    
        x = int(bbox[0])
        y = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])

        cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 255), 2)
        class_name = list_class_ids[i]
        image = cv2.putText(image, str(class_name), (x+2, y-3), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0))

        i += 1
    
    return image 
import cv2 
import numpy as np 
from shapely.geometry import Polygon
import math


def draw_box(image, bbox, color=(0,0,255)):
    # pts = np.array([[xs_af[0],ys_af[0]],[xs_af[1],ys_af[1]],[xs_af[2],ys_af[2]],[xs_af[3],ys_af[3]]], np.int32)
    pts = np.array(bbox)
    pts = pts.reshape((-1,1,2))
    image = cv2.polylines(image,[pts],True, color)

    return image 

def draw_list_bbox(image, list_bbox):
    for bbox in list_bbox:
        # print(bbox)
        image = draw_box(image, bbox)
    
    return image

def get_list_bbox(file_path):
    with open(file_path) as f:
        content = f.readlines()

    list_bbox = []
    list_bbox_char = []
    list_bbox_str = []
    content = [x.strip() for x in content]
    for ele in content:
        bbox_str = ele.split(" ",8)[-1]
        ele = ele.split()
        i = 0
        bbox = []
        while(i<8):
            xs = int(ele[i])
            ys = int(ele[i+1])
            point = [xs, ys]
            bbox.append(point)

            i += 2
        
        bbox_char = ele[8:]
        list_bbox_char.append(bbox_char)
        list_bbox.append(bbox)
        list_bbox_str.append(bbox_str)
    
    return list_bbox, list_bbox_char, list_bbox_str

def up_down_process(image, list_bbox):
    height, width, _ = image.shape
    max_y = 0
    min_y = 1e8

    list_space = np.linspace(0, height, num=5, dtype=int)
    # print("height: ", height)
    # print("list_space up down: ", list_space)
    for i in range(len(list_space)-1):
        _height = list_space[i+1]
        slide_poly = Polygon([[0, list_space[i]], [width, list_space[i]], [width, list_space[i]+1], [0, list_space[i+1]]])
        for j in range(len(list_bbox)):
            bbox = list_bbox[j]
            # print('bbox_{} in space {}->{}: {}'.format(j, list_space[i], list_space[i+1], bbox))
            text_poly = Polygon(bbox)
            if text_poly.contains(slide_poly) or text_poly.intersects(slide_poly):
                point_y1 = bbox[0][1]
                point_y2 = bbox[1][1]
                point_y3 = bbox[2][1]
                point_y4 = bbox[3][1]
                if i == 0 and j == 0:
                    min_y = (point_y1 + point_y2) // 2
                    max_y = (point_y3 + point_y4) // 2
                else:
                    if point_y1 <= min_y: min_y = point_y1 
                    if point_y2 <= min_y: min_y = point_y2
                    if point_y3 >= max_y: max_y = point_y3
                    if point_y4 >= max_y: max_y = point_y4
                
                # print('bbox_{} in space {}->{}: {}'.format(j, list_space[i], list_space[i+1], bbox))
    
    return min_y-10, max_y+10                    

def left_right_process(image, list_bbox):
    height, width, _ = image.shape
    max_x = 0
    min_x = 1e8

    list_space = np.linspace(0, width, num=5, dtype=int)
    for i in range(len(list_space)-1):
        _width = list_space[i+1]
        slide_poly = Polygon([[list_space[i], 0], [list_space[i+1], 0], [list_space[i]+1, height], [list_space[i], height]])
        for j in range(len(list_bbox)):
            bbox = list_bbox[j]
            text_poly = Polygon(bbox)
            if text_poly.contains(slide_poly) or text_poly.intersects(slide_poly):
                point_x1 = bbox[0][0]
                point_x2 = bbox[1][0]
                point_x3 = bbox[2][0]
                point_x4 = bbox[3][0]
                if i == 0 and j == 0:
                    min_x = (point_x1 + point_x2) // 2
                    max_x = (point_x3 + point_x4) // 2
                else:
                    if point_x1 <= min_x: min_x = point_x1  
                    if point_x2 <= min_x: min_x = point_x2
                    if point_x3 >= max_x: max_x = point_x3
                    if point_x4 >= max_x: max_x = point_x4
    
    return min_x-10, max_x+10

def create_vector(pointA, pointB):
    a = pointA[0] - pointB[0]
    b = pointA[1] - pointB[1]
    
    return (a, b)

def cal_dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def cal_mul(v1, v2):
    return 

def cal_length(v):
    return math.sqrt(cal_dotproduct(v, v))

def cal_angle(v1, v2):
    return math.acos(cal_dotproduct(v1, v2) / (cal_length(v1) * cal_length(v2)))

def cal_point_to_vector(point, p1, p2):
    point = np.array(point)
    p1 = np.array(p1)
    p2 = np.array(p2)
    d = np.linalg.norm(np.cross(p2-p1, p1-point))/np.linalg.norm(p2-p1)

    return d

def cal_distance_two_point(p1, p2):
    dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    return dist
    
def rotate_vector(point_rotation, point_tail, sin_a, cos_a):
    x = (point_tail[0] - point_rotation[0])*cos_a - (point_tail[1] - point_rotation[1])*sin_a + point_rotation[0]
    y = (point_tail[0] - point_rotation[0])*sin_a + (point_tail[1] - point_rotation[1])*cos_a + point_rotation[1]

    return [int(x), int(y)]

def get_info_rotation(list_bbox, point_par):
    bbox = list_bbox[0]
    point3 = (bbox[1][0], bbox[1][1])
    point4 = (bbox[2][0], bbox[2][1])

    vector_box = create_vector(point3, point4)

    vector_raw = create_vector(point3, point_par)

    # 3 canh tam giac vuong
    distance_par_vector = cal_point_to_vector(point_par, point3, point4)
    distance_p3_par = cal_distance_two_point(point3, point_par)
    edge_another = math.sqrt(distance_p3_par**2 - distance_par_vector**2)

    sin_a = distance_par_vector / distance_p3_par  
    cos_a = edge_another / distance_p3_par

    return point3, sin_a, cos_a



if __name__ == '__main__':
    annot_path = "image/annot.txt"
    image_path = "image/image.jpg"

    image = cv2.imread(image_path)
    list_bbox, list_bbox_char, list_bbox_str = get_list_bbox(annot_path)
    
    # xet thang nghieng thi xet song song la ok 
    # detect thang

    # up-down
    min_y, max_y = up_down_process(image, list_bbox)
    print("min_y: {}, max_y: {}".format(min_y, max_y))
    # left-right
    min_x, max_x = left_right_process(image, list_bbox)
    print("min_x: {}, max_x: {}".format(min_x, max_x))

    reciept_point1 = [min_x, min_y]
    reciept_point2 = [max_x, min_y]
    reciept_point3 = [max_x, max_y]
    reciept_point4 = [min_x, max_y]

    reciept_bbox = [reciept_point1 , reciept_point2, reciept_point3, reciept_point4]
    # image = draw_box(image, reciept_bbox)
    
    # cv2.imwrite("view_box_tmp.jpg", image)
 
    ########### 
    # detect nghieng

    # rotate vector
    root_head_point, sin_a, cos_a = get_info_rotation(list_bbox, [max_x, min_y])

    new_reciept_point1 = rotate_vector(root_head_point, reciept_point1, sin_a, cos_a)
    new_reciept_point2 = rotate_vector(root_head_point, reciept_point2, sin_a, cos_a)
    new_reciept_point3 = rotate_vector(reciept_point4, reciept_point3, sin_a, cos_a)
    root_tail_point = (root_head_point[0], reciept_point4[1])
    new_reciept_point4 = rotate_vector(root_tail_point, reciept_point4, sin_a, cos_a)

    # align point
    new_reciept_point2 = [new_reciept_point2[0]+20, new_reciept_point2[1]]
    new_reciept_box = [new_reciept_point1, new_reciept_point2, new_reciept_point3, new_reciept_point4]    
    print(new_reciept_box)

    # # visualize rotation
    # image = cv2.line(image, (new_reciept_point1[0], new_reciept_point1[1]), (new_reciept_point2[0], new_reciept_point2[1]), (255, 0, 0), 2)
    # image = cv2.line(image, (new_reciept_point4[0], new_reciept_point4[1]), (new_reciept_point3[0], new_reciept_point3[1]), (255, 0, 0), 2)

    image = draw_box(image, new_reciept_box)
    cv2.imwrite("view_box.jpg", image)
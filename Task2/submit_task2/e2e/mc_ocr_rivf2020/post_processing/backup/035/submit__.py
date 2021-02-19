import cv2
import numpy as np
import re
import pandas as pd
import os
from collections import deque  
from sklearn.metrics.pairwise import cosine_similarity
import collections
from math import sqrt
from unidecode import unidecode
from tqdm import tqdm
from recognizers_text import Culture, ModelResult
from recognizers_date_time import DateTimeRecognizer
from .create_prices_proprocess_json import PRICES_PREPROCESS, PRICES_CHAR, PREFIX_CHAR, ADDRESS_PREPROCESS,\
                                         TIME_PREPROCESS, PREFIX_PRIORITIZE, ADDRESS_POSTPROCESS, SELLER_POSTPROCESS
from .sellerMatch import sellerMatch, output_Prices_Match, prefixMatch, addressMatch, findAddress, prefixPreprocess, findSeller
from shapely.geometry import Polygon

with open("post_processing/field_dictionary/street.txt") as f:
    content = f.readlines()
LIST_STREET_DEF = [x.strip() for x in content] 

with open("post_processing/field_dictionary/seller.txt") as f:
    content = f.readlines()
LIST_SELLER_DEF = [x.strip() for x in content] 

with open("post_processing/field_dictionary/phone.txt") as f:
    content = f.readlines()
LIST_PHONE_DEF = [x.strip() for x in content] 

with open("post_processing/field_dictionary/prices.txt") as f:
    content = f.readlines()
LIST_PRICES_DEF = [x.strip() for x in content] 

with open("post_processing/field_dictionary/prices_prioritize.txt") as f:
    content = f.readlines()
LIST_PRICES_PRIORITIZE_DEF = [x.strip() for x in content] 

with open("post_processing/field_dictionary/date.txt") as f:
    content = f.readlines()
LIST_DATE_DEF = [x.strip() for x in content] 

model = DateTimeRecognizer(Culture.English).get_datetime_model()

def postprocessTimestamp(raw_input):
    raw_input = raw_input.split('|||')
    print(raw_input)
    if len(raw_input) == 1:
        return extractTimestamp(raw_input[0])
        # return raw_input[0]
    res = []
    day_symbol = ['/','-','.']
    has_colon = False
    for component in raw_input:
        bad_word = ['HET HAN','LAM VIEC','CODE']
        isbad = False

        for x in re.findall('[0-9]+', component):
            if len(x) >= 5:
                isbad = True
                break
        for word in bad_word:
            if unidecode(component.upper()).find(word) != -1:
                isbad = True
                break
        if isbad:
            continue
        component = extractTimestamp(component) 
        res.append(component)
        if component.find(':') != -1:
            has_colon = True
    print("res: ", res) 
  
    
    if has_colon == False:
        return res[0]

    for x in res:
        tmp = [x.count(y) for y in day_symbol if x.find(y)!=-1]
        tmp1 = [1 for y in x.split() if any(map(str.isdigit, y))]
        print(x,x.split(),tmp1)
        if x.find(':') != -1 and len(tmp1) > 1 and len(tmp) > 0 and sum(tmp) >= 2:
            return x
    
    has_colon =  False
    has_day = False
    for i, x in enumerate(res):
        if x.find(':') != -1:
            has_colon = True
        if len([1 for y in day_symbol if x.find(y)!=-1]) > 0:
            has_day = True
        print("has_day: {}, has_colon: {}".format(has_day, has_colon))
        if has_colon and has_day:
            return ' '.join(res[i-1:i+1])

def extractTimestamp(raw_input):
    res = model.parse(raw_input)
    res = [x for x in res if (len(x.text) != 4 or (not x.text.isdigit))]
    res = [x for x in res if x.text!='thu' and (x.text[-1]!='p' or x.text[:-1].isdigit()==False)]
    if len(res) == 0:
        return None
    # for x in res:
    #   print(x.text)
    st = res[0].start
    en = res[-1].end 
    result = raw_input[st : en + 1]
    if any(map(str.isdigit, result)) == False:
        return None
    
    prefix = raw_input[:st].split()
    suffix = raw_input[en + 1:].split()

    if len(prefix):
        if any(map(str.isdigit, prefix[-1])):
            result = prefix[-1] + ' ' + result
            prefix = prefix[:-1]
        ind = -1
        if abs(ind) <= len(prefix) and (unidecode(prefix[ind].upper()) == 'NGAY' or prefix[ind][-1] == ':'):
            while abs(ind) < len(prefix) and prefix[ind][0].islower() or prefix[ind] == ':':
                ind -= 1
            result = ' '.join(prefix[ind:]) + ' ' + result

    if len(suffix):
        if any(map(str.isdigit, suffix[0])):
            result = result + ' ' + suffix[0]
            suffix = suffix[1:]
        ind = 0
        if ind < len(suffix) and suffix[ind][0] == '(':
            while ind + 1 < len(suffix) and suffix[ind][-1] != ')':
                ind += 1
            result = result + ' ' + ' '.join(suffix[:ind])
    
    for x in re.findall('[0-9]+', result):
        if len(x) >= 5:
            result = result.replace(x,'')

    garbage = ['thu','  ']
    for x in garbage:
        result = result.replace(x,'')

    return result.replace('thu','').strip() if len(result) > 4 else None

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

def search_day(input_string):
    result = re.search("^([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])(\.|-|/)([1-9]|0[1-9]|1[0-2])(\.|-|/)([0-9][0-9]|19[0-9][0-9]|20[0-9][0-9])$|^([0-9][0-9]|19[0-9][0-9]|20[0-9][0-9])(\.|-|/)([1-9]|0[1-9]|1[0-2])(\.|-|/)([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])$", input_string)
    if result == None:
        result = re.search("^\d\d\d\d/(0?[1-9]|1[0-2])/(0?[1-9]|[12][0-9]|3[01]) (00|[0-9]|1[0-9]|2[0-3]):([0-9]|[0-5][0-9]):([0-9]|[0-5][0-9])$", input_string)

    return result

def get_center(bbox):
    center_x = (bbox[0][0] + bbox[2][0]) // 2
    center_y = (bbox[0][1] + bbox[2][1]) // 2
    return (center_x, center_y)

def draw_line(image, start_point, end_point):
    image = cv2.line(image, start_point, end_point, (0, 255, 255), 2)

    return image

def get_vector(point1, point2):
    vector = np.array([point2[0] - point1[0], point2[1] - point1[1]])
    vector = vector.reshape(1, 2)

    return vector

def compute_cosine(vector_bbox_prices, vector_gt):
    cosine = cosine_similarity(vector_bbox_prices, vector_gt)
    cosine = float(cosine[0][0])
    
    return cosine

def get_index_cosine(list_cosine, list_index_cosine):
    min_cosine = np.amax(list_cosine)
    index = np.where(list_cosine == min_cosine)
    index = index[0][0]
    index = list_index_cosine[index]
    return index

def get_day(list_bbox_char, list_bbox_str):
    list_number_prices = []
    index = None
    for i in range(len(list_bbox_char)):
        if len(list_bbox_str[i]) < 12: # > 100ty
            list_bbox_str[i] = list_bbox_str[i].replace(",", ".")
            list_number_prices.append(i)
            
        if list_bbox_str[i].find(":") != -1:
            value = list_bbox_str[i].split(":")[1]
            if len(value) < 10: 
                list_number_prices.append(i)

        bbox = list_bbox_char[i]
        for content in bbox:
            result = search_day(content)
            if result != None:
                result = result.group()
                # print(list_bbox_str[i])
                index = i 
    
    # chay cua cuong` 
    res = []
    flag = False
    for i in range(len(list_bbox_char)):
        bbox = list_bbox_char[i]
        for content in bbox:
            result = extractTimestamp(content)
            if result != None:
                flag = True
                if i not in res:
                    res.append(i)
    
    # cuong_result = '|||'.join(res)

    return list_number_prices, index, flag, res

def get_top_prices(list_number_prices, list_bbox_str, top=5):
    prices_top = deque([], top) # lấy top 1 trước có gì rồi sửa lại lấy top k 
    backup_arr = {}
    max_number = 0.0
    print(list_bbox_str)
    list_char_prices = ['d', 'đ', 'Đ', 'D', 'đồng']
    check_has_onecolumn = False
    true_index = 0
    for i in list_number_prices:
        # prices = 0
        print("i: ", i)
        print("list_bbox_str[i]: ", list_bbox_str[i])
        try:
            # if list_bbox_str[i].find('.') == -1:
            #     continue
            
            prices = list_bbox_str[i]
            # list_monney = ["UND, VND, UNĐ, VNĐ, und, unđ, vnđ"]
        
            prices = prices.replace("UND", "")
            prices = prices.replace("VND", "")
            prices = prices.replace("UNĐ", "")
            prices = prices.replace("VNĐ", "")
            prices = prices.replace("und", "")
            prices = prices.replace("unđ", "")
            prices = prices.replace("vnđ", "")

            print("PRICES: ", prices)
            # check 1 column
            if prices.find(":") != -1:
                prices = prices.split(":")
                print("prices in 1 column: ", prices)
                prefix = prices[0]
                value = prices[1]
                print("type value: {}, type prefix: {}".format(type(value), type(prefix)))
                for char_price in list_char_prices:
                    if value.find(char_price) != -1:
                        value = value.replace(char_price, '')
                    index = value.find(" ")

                    if index != -1:
                        value = value.replace(" ", ".")
                    
                    if value.find(",") != -1:
                        value = value.replace(",", "")

                    if value.find(".", index+1) != -1:
                        value = value.replace(".", "")
                    
                    
                value = float(value)
                print("value: ", value)
                
                if value >=50.0:
                    # if prefix in 2 two list => return
                    if prefix in LIST_PRICES_PRIORITIZE_DEF:    
                        check_has_onecolumn = True
                        true_index = i
                        return prices_top, check_has_onecolumn, true_index

                    if prefix in LIST_PRICES_DEF:
                        check_has_onecolumn = True
                        true_index = i
                        return prices_top, check_has_onecolumn, true_index

            print("prices: ", prices)
            for char_price in list_char_prices:
                if prices.find(char_price) != -1:
                    prices = prices.replace(char_price, '')
                index = prices.find(" ")
                if index != -1:
                    prices = prices.replace(" ", ".")
                if prices.find(".", index+1) != -1:
                    prices = prices.replace(".", "")
                        

            prices = float(prices)

            if prices >= 500.0 and prices < 1000000000.0:
                if prices >= max_number:
                    max_number = prices
                    prices_top.append(i)
                    print("prices_top: ", prices_top)
                else:
                    if prices > 0:
                        backup_arr[i] = prices
        except Exception as e:
            print("bug in get top prices: ", e)
            pass # string not number!
    
    print("prices top before append phamn bu`: ", prices_top)
    # them phan bu` cho du top5
    backup_arr = {k: v for k, v in sorted(backup_arr.items(), key=lambda item: item[1], reverse=True)}
    print("backup_arr: ", backup_arr)
    print("len backup arr: ", len(backup_arr))
    number_missed = top - len(prices_top)
    print("number_missed: ", number_missed)
    if number_missed > 0:
        i = 0
        for key, value in backup_arr.items():
            if i <= number_missed:
                prices_top.append(key)
                i += 1

    print("prices top: ", prices_top)

    return prices_top, check_has_onecolumn, true_index

def compute_distance(pointA, pointB):
    dist = sqrt( (pointB[0] - pointA[0])**2 + (pointB[1] - pointA[1])**2 )

    return dist

def get_index_dist(list_dist, list_index_dist):
    min_cosine = np.amin(list_dist)
    index = np.where(list_dist == min_cosine)
    index = index[0][0]
    index = list_index_dist[index]

    return index

def get_street(height_img, width_img, name_box, list_bbox, list_index_street):
    list_dist = []
    list_index_dist = []
    vector_gt = get_vector((0, height_img), (width_img, height_img))
    center_name_box = get_center(name_box)
    for i in list_index_street:
        bbox = list_bbox[i]
        center_bbox = get_center(bbox)
        dist = compute_distance(center_bbox, center_name_box)
    
        list_dist.append(dist)
        list_index_dist.append(i)
    
    bbox_index = get_index_dist(list_dist, list_index_dist)
    print("bbox index cosine street: ", bbox_index)

    return bbox_index

def get_prices(height_img, width_img, prices_box, list_bbox, list_bbox_str):
    # check parallel 
    list_cosine = []
    list_index_cosine = []
    vector_gt = get_vector((0, height_img), (width_img, height_img))
    center_prices_bbox = get_center(prices_box)
    for i in range(len(list_bbox)):
        bbox = list_bbox[i]
        center_bbox = get_center(bbox)
        vector_bbox_prices = get_vector(center_bbox, center_prices_bbox)
        cosine = compute_cosine(vector_bbox_prices, vector_gt)
        list_cosine.append(cosine)
        list_index_cosine.append(i)
        # image = draw_line(image, center_box, center_prices_box)
        # image = draw_box(image, bbox, (255, 255, 0))
        
    bbox_index = get_index_cosine(list_cosine, list_index_cosine)

    return bbox_index

def get_index_street(list_bbox_str, number_line=7):
    list_street = []
    print("Get index_streeet")
    for i in range(number_line):
        content = list_bbox_str[i].lower()
        new_content = findAddress(content)
        print(new_content)
        for word in LIST_STREET_DEF:
            if content.find(word) != -1:
        # if new_content != None:
                flag = False
                for char in LIST_PHONE_DEF:
                    if content.find(char) != -1:
                        flag = True 
                        break 
                if flag == False:
                    if i not in list_street:
                        # print("content in list street: ", content)
                        list_street.append(i)
                        break
    print("list street: ", list_street)
    return list_street

# format bbox:  [[66, 1222], [67, 1195], [369, 1209], [368, 1236]]
def get_line_box(bbox, height, width):
    center_bbox = get_center(bbox)
    point1 = None
    point2 = None
    point3 = None
    point4 = None
    for point in bbox:
        if point[0] < center_bbox[0] and point[1] < center_bbox[1]:
            point1 = [0, point[1]]
        elif point[0] > center_bbox[0] and point[1] < center_bbox[1]:
            point2 = [width, point[1]]
        elif point[0] < center_bbox[0] and point[1] > center_bbox[1]:
            point3 = [0, point[1]]
        else:
            point4 = [width, point[1]]
    
    return [point1, point2, point3, point4]

def check_two_box_intersec(bbox1, bbox2, height, width):
    line_bbox1 = get_line_box(bbox1, height, width)
    line_bbox2 = get_line_box(bbox2, height, width)

    polygon_bbox1 = Polygon(line_bbox1)
    polygon_bbox2 = Polygon(line_bbox2)

    if polygon_bbox1.intersects(polygon_bbox2):
        return True
    
    return False

def get_index_name(list_bbox_str, number_line=6):
    # for i in range(number_line):
    #     flag = False
    #     content = list_bbox_str[i].lower()
    #     print("content in get index name: ", content)
    #     for word in LIST_STREET_DEF:
    #         if content.find(word) != -1:
    #             flag = True
    #             break
        
    #     if flag == False:
    #         print("index name: ", i)
    #         return i

    # seller cua Hung`
    for i in range(number_line):
        flag = False
        content = list_bbox_str[i].lower()
        print("content in get index name: ", content)
        for word in LIST_SELLER_DEF:
            if content.find(word) != -1:
        # new_content = findSeller(content)
                # if new_content != None:
                flag = True
                break
        
        if flag == True:
            print("index name: ", i)
            return i
    
    for i in range(number_line):
        # flag = False
        content = list_bbox_str[i].lower()
        new_content = findSeller(content)
        if new_content != None:
           return i
                
    
    return None

def get_submit_image(image_path, annot_path):
    output_dict = {}
    image = cv2.imread(image_path)

    height, width, channels = image.shape 

    list_bbox, list_bbox_char, list_bbox_str = get_list_bbox(annot_path)
    
    # get day
    list_number_prices, index_day_bbox, flag, cuong_result= get_day(list_bbox_char, list_bbox_str)
    # day = list_bbox_str[index_day_bbox]
    # print(index_day_bbox)
    if flag == True:
        try:
            list_output_time = []
            print("CUong result: ", cuong_result)
            check_len_list_day = 0
            for index_time in cuong_result:
                print("NUMBER INDEX: ", index_time)
                day = list_bbox_str[index_time]
                day = day.split()
                print("day before: ", day)
                for key, value in TIME_PREPROCESS.items():
                    print("key: {}, value: {}".format(key, value))
                    for ele in value:
                        for i in range(len(day)):
                            char = day[i]
                            print("char: ", char)
                            if char == ele:
                                print("day[i] before: ", day[i])
                                day[i] = key
                                print("day[i] after: ", day[i])
                                print(day)
                                break
                
                day = ' '.join(map(str, day))
                print("DAY before post process", day)

                list_time = day.split()
                list_block_re_extract = ["Ngày", "bán"]

                check_block = False
                for my_time in list_time:
                    if my_time in list_block_re_extract:
                        check_block = True 
                        break 
                
                if check_block == False:
                    if len(list_time) > 3:
                        print("len list time: ", len(list_time))
                        print("list_time: ", list_time)
                        print("day before re extracttime: ", day)
                        day = extractTimestamp(day)

                day = day.replace(".", ":")

                # day_tmp = day.lower()
                # print("DAY tmp before post process", day_tmp)

                # if "thời" not in day_tmp or "print" not in day_tmp:
                #     print("Don't have thoi")
                #     day = postprocessTimestamp(day)

                # print("DAY after post process: ", day)

                # remove long string not have [",", ":", "/"]
                # day = day.split()
                # list_char_time = ["/", ":"]
                # for ele in day:
                #     if len(ele) > 15:
                #         day.remove(ele)
                #     else:
                #         flag = False
                #         for char_time in list_char_time:
                #             if ele.find(char_time) == -1:
                #                 flag = True
                #             else:
                #                 flag = False
                #                 break
                        
                #         if flag == True:
                #             day.remove(ele)

                # day = ' '.join(map(str, day))
                ############# 

                print("NUMBER INDEX: ", index_time)
                print("day after append: ", day)
                if check_len_list_day > 2:
                    break
                    
                list_output_time.append([index_time, day])

            list_output_time.sort(key = lambda x: get_center(list_bbox[x[0]]))
            print("BBOX: ", list_bbox[list_output_time[0][0]])
            if len(list_output_time) == 3:
                if check_two_box_intersec(list_bbox[list_output_time[0][0]], list_bbox[list_output_time[1][0]], height, width):
                    if check_two_box_intersec(list_bbox[list_output_time[0][0]], list_bbox[list_output_time[2][0]], height, width) == False:
                        list_output_time.pop(2)
                else:
                    if check_two_box_intersec(list_bbox[list_output_time[1][0]], list_bbox[list_output_time[2][0]], height, width) == True:
                        list_output_time.pop(0)

            print("LIST OUTPUT TIME: ", list_output_time)
            cnt = 0
            for output_time in list_output_time:
                output_dict[331+cnt] = [output_time[1], 'TIMESTAMP']
                check_len_list_day += 1
                cnt += 1
        except Exception as e:
            print(e)
            print("Not found index!")
            pass
    else:
        try:
            output_dict[index_day_bbox] = [list_bbox_str[index_day_bbox], 'TIMESTAMP']
        except:
            print("Not found index!")
            pass
    
    # get prices
    top_number = 20
    prices_top, check_has_onecolumn, true_index = get_top_prices(list_number_prices, list_bbox_str, top_number)
    if check_has_onecolumn:
        print("Has 1 column")
        prices = list_bbox_str[true_index].split(":")
        prefix_raw = prices[0]
        prices_value = prices[1]
        print("prices: ", prices)

        # preprocess value
        tmp = False
        for key, value in PRICES_CHAR.items():
            for ele in value:
                index = prices_value.find(ele)
                if index != -1:
                    tmp = True
                    prices_value = prices_value.replace(ele, key)
                    print("prices_value: ", prices_value)
                    break

            if tmp == True:
                break
        
        # preprocess prefix
        list_prefix = prefix_raw.split()
        # print("list prefix: ", list_prefix)
        for key, value in PRICES_PREPROCESS.items():
            for ele in value:
                for i in range(len(list_prefix)):
                    char = list_prefix[i]
                    if char == ele:
                        list_prefix[i] = key
                        break
        
        tmp = False
        prefix_raw = ' '.join(map(str, list_prefix))
        print("prefix_raw: ", prefix_raw)
        for key, value in PREFIX_CHAR.items():
            for ele in value:
                index = prefix_raw.find(ele)
                if index != -1:
                    tmp = True
                    prefix_raw = prefix_raw.replace(ele, key)
                    break
                    
            if tmp == True:
                break

        prefix_raw_preprocess = prefixPreprocess(prefix_raw)
        if prefix_raw_preprocess != 'Nomatching':
            prefix_raw = prefix_raw_preprocess
        print("prefix_raw: ", prefix_raw_preprocess)
        prices = prefix_raw + '|||' + prices_value
        output_dict[777] = [prefix_raw, 'TOTAL_COST']
        output_dict[778] = [prices_value, 'TOTAL_COST']

    else:
        try:
            # check trong list uu tien 
            flag_found = False
            list_output_prices = []
            for i in range(len(prices_top)):
                index_prices = prices_top[i]
                prices_box = list_bbox[index_prices]
                prices = list_bbox_str[index_prices]
                index_string_prices = get_prices(height, width, prices_box, list_bbox, list_bbox_str)
                
                prefix_raw = list_bbox_str[index_string_prices]
                prefix = prefix_raw.lower()
                #  kiem tra list prices uu tien
                flag = False
                print("prefix: ", prefix)
                prefix = prefixMatch(prefix)
                print("prefix after: ", prefix)
                print("prices: ", list_bbox_str[index_prices])
                if prefix != None:
                    for word in LIST_PRICES_PRIORITIZE_DEF:
                        # print("prefix in PRIORITIZE: ", prefix)
                        if prefix.find(word) != -1:
                            flag = True
                            break
                
                if flag==True:
                    # preprocess
                    tmp = False
                    prices_value = list_bbox_str[index_prices]
                    for key, value in PRICES_CHAR.items():
                        for ele in value:
                            index = prices_value.find(ele)
                            if index != -1:
                                tmp = True
                                prices_value = prices_value.replace(ele, key)
                                print("prices_value: ", prices_value)
                                break

                        if tmp == True:
                            break

                    print("########")
                    print("prefix: ", prefix_raw)
                    print("########")
                    list_prefix = prefix_raw.split()
                    for key, value in PRICES_PREPROCESS.items():
                        for ele in value:
                            for i in range(len(list_prefix)):
                                char = list_prefix[i]
                                if char == ele:
                                    list_prefix[i] = key
                                    break
                    
                    tmp = False
                    prefix_raw = ' '.join(map(str, list_prefix))
                    for key, value in PREFIX_CHAR.items():
                        for ele in value:
                            index = prefix_raw.find(ele)
                            if index != -1:
                                tmp = True
                                prefix_raw = prefix_raw.replace(ele, key)
                                print("prefix: ", prefix_raw)
                                break
                                
                        if tmp == True:
                            break
                        
                    print("index prices: ", index_prices)
                    # prices = prefix_raw + '|||' + prices_value
                    prefix_raw_preprocess = prefixPreprocess(prefix_raw)
                    if prefix_raw_preprocess != 'Nomatching':
                        prefix_raw = prefix_raw_preprocess

                    print("PREFIX RAW: ", prefix_raw)
                    if prefix_raw == 'Tiền mặt':
                        prefix_raw = 'Tổng'
                    if prefix_raw == 'VNĐ':
                        prefix_raw = 'Tổng cộng'

                    list_output_prices.append((prefix_raw, prices_value))
            
            print("list_output_prices: ", list_output_prices)    
            list_result = []
            for output_prices in list_output_prices:
                key = output_prices[0]
                if key == 'Tổng số:' or key == 'Tổng số':
                    continue
                value = output_prices[1]
                number = output_Prices_Match(key)
                if len(key) >= 5:
                    list_result.append([number, key, value])
            print("list_result: ", list_result)

            # sort list_result nua roi lay phan tu dau tien
            list_result.sort(key = lambda x: x[0])
            print("list_result after sort: ", list_result)
            result_prices = list_result[0]
            
            output_dict[777] = [result_prices[1], 'TOTAL_COST']
            output_dict[778] = [result_prices[2], 'TOTAL_COST']
            flag_found = True
            
            if flag_found == False:
                for i in range(len(prices_top)):
                    index_prices = prices_top[i]
                    prices_box = list_bbox[index_prices]
                    prices = list_bbox_str[index_prices]
                    index_string_prices = get_prices(height, width, prices_box, list_bbox, list_bbox_str)
                    
                    prefix_raw = list_bbox_str[index_string_prices]
                    prefix = prefix_raw.lower()
                    #  kiem tra list prices uu tien
                    flag = False
                    print("prefix: ", prefix)
                    if prefixMatch(prefix) != None:
                        prefix = prefixMatch(prefix)
                    print("prefix after: ", prefix)
                    print("prices: ", list_bbox_str[index_prices])
                    if prefix != None:
                        for word in LIST_PRICES_PRIORITIZE_DEF:
                            # print("prefix in PRIORITIZE: ", prefix)
                            if prefix.find(word) != -1:
                                flag = True
                                break
                    
                    if flag==True:
                        # preprocess
                        tmp = False
                        prices_value = list_bbox_str[index_prices]
                        for key, value in PRICES_CHAR.items():
                            for ele in value:
                                index = prices_value.find(ele)
                                if index != -1:
                                    tmp = True
                                    prices_value = prices_value.replace(ele, key)
                                    print("prices_value: ", prices_value)
                                    break

                            if tmp == True:
                                break

                        print("########")
                        print("prefix: ", prefix_raw)
                        print("########")
                        list_prefix = prefix_raw.split()
                        for key, value in PRICES_PREPROCESS.items():
                            for ele in value:
                                for i in range(len(list_prefix)):
                                    char = list_prefix[i]
                                    if char == ele:
                                        list_prefix[i] = key
                                        break
                        
                        tmp = False
                        prefix_raw = ' '.join(map(str, list_prefix))
                        for key, value in PREFIX_CHAR.items():
                            for ele in value:
                                index = prefix_raw.find(ele)
                                if index != -1:
                                    tmp = True
                                    prefix_raw = prefix_raw.replace(ele, key)
                                    print("prefix: ", prefix_raw)
                                    break
                                    
                            if tmp == True:
                                break
                            
                        print("index prices: ", index_prices)
                        prefix_raw_preprocess = prefixPreprocess(prefix_raw)
                        if prefix_raw_preprocess != 'Nomatching':
                            prefix_raw = prefix_raw_preprocess
                        print("PREFIX RAW: ", prefix_raw)
                        # prices = prefix_raw + '|||' + prices_value
                        if prefix_raw == 'Tiền mặt':
                            prefix_raw = 'Tổng'
                        if prefix_raw == 'VNĐ':
                            prefix_raw = 'Tổng cộng'
                            
                        list_output_prices.append((prefix_raw, prices_value))
                
                print("list_output_prices: ", list_output_prices)    
                list_result = []
                for output_prices in list_output_prices:
                    key = output_prices[0]
                    if key == 'Tổng số:' or key == 'Tổng số':
                        continue
                    value = output_prices[1]
                    number = output_Prices_Match(key)
                    if len(key) >= 5:
                        list_result.append([number, key, value])
                print("list_result: ", list_result)

                # sort list_result nua roi lay phan tu dau tien
                list_result.sort(key = lambda x: x[0])
                print("list_result after sort: ", list_result)
                result_prices = list_result[0]
                
                output_dict[777] = [result_prices[1], 'TOTAL_COST']
                output_dict[778] = [result_prices[2], 'TOTAL_COST']

        except Exception as e:
            print(e)
            print("Not found index!")
            pass

    # get name
    index_name = get_index_name(list_bbox_str)
    # print(index_name)
    name_bbox = None
    try:
        list_name = list_bbox_str[index_name]
        if sellerMatch(list_name) != None:
            list_name = sellerMatch(list_name)
        print(list_name)
        if list_name == 'Vin Commerce':
            list_name = 'VinCommerce'
        # print("list_name: ", list_name)
        # list_name = list_name.split()
        # for key, value in SELLER_POSTPROCESS.items():
        #     print("key: {}, value: {}".format(key, value))
        #     for ele in value:
        #         for i in range(len(list_name)):
        #             char = list_name[i]
        #             if char == ele:
        #                 list_name[i] = key
        #                 print("key: ", key)
        #                 break
        
        # list_name = ' '.join(map(str, list_name))
        name_bbox = list_bbox[index_name]
        if "site" in list_name:
            list_name = ""
        output_dict[0] = [list_name, 'SELLER']
    except:
        print("Not found index!")
        pass

    # get street
    try:
        list_index_street = get_index_street(list_bbox_str)
        print("list index street: ", list_index_street)
        
        # cnt = 0 
        # for index_street in list_index_street:
        #     list_street = list_bbox_str[index_street]
        #     list_street = list_street.split()
        #     print("list street: ", list_street)
        #     for key, value in ADDRESS_PREPROCESS.items():
        #         for ele in value:
        #             for i in range(len(list_street)):
        #                 char = list_street[i]
        #                 if char == ele:
        #                     list_street[i] = key
        #                     break

        #     list_street = ' '.join(map(str, list_street))
        
        print("len(list_index_street): ", len(list_index_street))

        if len(list_index_street) == 1:
            list_street = list_bbox_str[list_index_street[0]]
            list_street = list_street.split()
            print("list street: ", list_street)

            # post process
            for key, value in ADDRESS_POSTPROCESS.items():
                for ele in value:
                    for i in range(len(list_street)):
                        char = list_street[i]
                        if char == ele:
                            list_street[i] = key
                            break
            
            # preprocess
            for i in range(len(list_street)):
                street = list_street[i]
                print(street)
                street = addressMatch(street)
                if street != None:
                    list_street[i] = street
                
            list_street = ' '.join(map(str, list_street))

            if "Enail" in list_street:
                list_street = ""
            if "Email" in list_street:
                list_street = ""

            output_dict[250] = [list_street, 'ADDRESS']
               
        else:
            # remove seller in list_index_street
            for i in list_index_street:
                print("list_bbox_str[i]: ", list_bbox_str[i])
                if list_bbox_str[i] == list_bbox_str[index_name]:
                    list_index_street.remove(i)

            index = get_street(height, width, name_bbox, list_bbox, list_index_street)
            list_street = list_bbox_str[index]
            list_street = list_street.split()
            print("list street: ", list_street)
            # post process
            for key, value in ADDRESS_POSTPROCESS.items():
                for ele in value:
                    for i in range(len(list_street)):
                        char = list_street[i]
                        if char == ele:
                            list_street[i] = key
                            break
            
            # preprocess
            for i in range(len(list_street)):
                street = list_street[i]
                street = addressMatch(street)
                if street != None:
                    list_street[i] = street

            list_street = ' '.join(map(str, list_street))
            if "Enail" in list_street:
                list_street = ""
            if "Email" in list_street:
                list_street = ""
            print("list_street final: ", list_street)
            output_dict[250] = [list_street, 'ADDRESS']

    except Exception as e:
        print(e)
        print("Not found index!")
        pass

    # sort
    output_dict = collections.OrderedDict(sorted(output_dict.items()))

    return output_dict

def print_output(output_dict):
    list_value = []
    list_field = []
    for key, value in output_dict.items():
        list_value.append(value[0])
        list_field.append(value[1])
    
    print(list_value)
    print(list_field)

    result_value = '|||'.join(list_value)
    result_field = '|||'.join(list_field)
    
    return result_value, result_field

def get_output(output_dict):
    list_value = []
    list_field = []
    for key, value in output_dict.items():
        list_value.append(value[0])
        list_field.append(value[1])
    
    # result_value = '|||'.join(list_value)
    # result_field = '|||'.join(list_field)
    
    return list_value, list_field

def create_result(task1_csv_path='results.csv'):
    # -------------------------
    # Argument
    # task1_csv_path: path of task 1 csv
    # ------------------------- 
    dtf = pd.read_csv(task1_csv_path)
    field_rank = {
        'SELLER' : 1,
        'ADDRESS' : 2,
        'TIMESTAMP' : 3,
        'TOTAL_COST' : 4
    }

    res = []
    for key, row in dtf.iterrows():
        name = row['img_id']
        name = name.split(".")[0]
        annot_path = os.path.join('result_txt', name+".txt")
        image_path = os.path.join('upload', name+".jpg")

        output_dict = get_submit_image(image_path, annot_path)
        field_value, field_name = get_output(output_dict)
        print(field_name)
        print(field_value)
        field = list(zip(field_value, field_name))
        field.sort(key = lambda x: field_rank[x[1]])
        res.append('|||'.join([str(x[0]) for x in field]))

    dtf['anno_texts'] = res
    if not os.path.exists("submit"):
        os.mkdir("submit")

    pd.to_csv(os.path.join('submit', 'results.csv'))

if __name__ == "__main__":
    # submit
        # create_result()

    name = "mcocr_private_145121pgxpu"

    annot_path = os.path.join('result_txt', name+".txt")
    image_path = os.path.join('upload', name+".jpg")

    output_dict = get_submit_image(image_path, annot_path)
    print(output_dict)
    result_value, result_field = print_output(output_dict)
    # print(output_dict)
    
    print(result_field)
    print(result_value)
 

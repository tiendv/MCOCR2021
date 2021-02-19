import os 
import pandas as pd 
import shutil 
import json
import ast
import lmdb  # install lmdb by "pip install lmdb"
import cv2
import numpy as np
import glob
from PIL import Image, ImageDraw, ImageFont


INPUT_FOLDER_IMAGES = '/home/huy/TEST_SUBMIT/MC-OCR-RIVF2020/mc-ocr_rivf2020/mcocr_public_train_test_shared_data/mcocr_train_data/train_images'
INPUT_FOLDER_IMAGES_CROP = '/home/huy/TEST_SUBMIT/MC-OCR-RIVF2020/mc-ocr_rivf2020/pre_processing/crop_xoay/output_deskew_train'
INPUT_FOLDER_TXT = '/home/huy/TEST_SUBMIT/MC-OCR-RIVF2020/mc-ocr_rivf2020/result_txt_mcocr_train_data_crop'
OUTPUT_VISUALIZE = 'visualize_full/API'
OUTPUT_VISUALIZE_BOX_GT = 'visualize_full/bbox'
OUTPUT_VISUALIZE_LINE = 'visualize_full/line'
OUTPUT_VISUALIZE_MERGE = 'visualize_full/merge'
OUTPUT_VISUALIZE_POLYGON = 'visualize_full/API_polygon'

def get_boxs(txt_path):
    with open(txt_path) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    start_points = []
    end_points =[]
    words = []
    for ele in content:
        bbox_str = ele.split(" ")[0:8]
        world = ele.split(" ",8)[-1]
        box = [ int(b) for b in bbox_str]
        # print(box)
        if box[5] > box[1]:
            start_point = (int(box[0])-10, int(box[1])-20)
            end_point = (int(box[4]), int(box[5]))
        else:
            start_point = (int(box[2])-10, int(box[3])-20)
            end_point = (int(box[6]), int(box[7]))
        # print(start_point)
        start_points.append(start_point)
        end_points.append(end_point)
        words.append(world)
    return start_points, end_points, words

def visualize(img ,img_path, txt_path, output):

    img_1 = np.ones([img.shape[0],img.shape[1],3],dtype=np.uint8)*255
    fontpath = "visualize/unicode.publish.UVNVietSach_R.TTF" 
    
    im = Image.open(img_path) # img1 anh task 1
    draw1 = ImageDraw.Draw(im)

    img_pil = Image.fromarray(img_1)
    draw = ImageDraw.Draw(img_pil)
    b,g,r,a = 0,0,255,0
    name_img = img_path.split('/')[-1]
    # txt_name = name_img.replace('.jpg', '.txt')
    # txt_path = os.path.join(txt_folder, txt_name)
    print(txt_path)
    start_points, end_points, words =  get_boxs(txt_path)
    for start_point,end_point, word in zip(start_points, end_points, words):
        font_size = abs(end_point[1] - start_point[1])//2
        print(font_size)
        font = ImageFont.truetype(fontpath, font_size)
        draw.text(start_point,  word, font = font, fill = (b, g, r, a))
        draw1.rectangle([start_point,end_point], outline ="red") 
        draw.rectangle([start_point,end_point], outline ="red") 
        img_2 = np.array(img_pil) # img2 anh task 2
    # cv2.imwrite('task2_'+str(name_img),img_2)
    new_image = Image.new('RGB',(2*im.size[0], im.size[1]), (250,250,250))
    new_image.paste(im,(0,0))
    new_image.paste(img_pil,(im.size[0],0))
    print(output)
    if not os.path.exists(output):
        os.mkdir(output)
    output_path_file = os.path.join(output, name_img)
    img_pil.save(output_path_file,"JPEG")

def draw_bbox(img_path, result, color=(255, 0, 0), thickness=2):
    if isinstance(img_path, str):
        img_path = cv2.imread(img_path)
        # img_path = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
    img_path = img_path.copy()
    for point in result:
        # point = point.astype(int)
        cv2.line(img_path, tuple(point[0]), tuple(point[1]), color, thickness)
        cv2.line(img_path, tuple(point[1]), tuple(point[2]), color, thickness)
        cv2.line(img_path, tuple(point[2]), tuple(point[3]), color, thickness)
        cv2.line(img_path, tuple(point[3]), tuple(point[0]), color, thickness)
    return img_path

def get_boxs_list(txt_path):
    with open(txt_path) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    boxes_list = []
    for ele in content:
        bbox_str = ele.split(" ")[0:8]
        world = ele.split(" ",8)[-1]
        bbox = []
        for i in range(0, 8, 2):
            xs = int(bbox_str[i])
            ys = int(bbox_str[i+1])
            point = [xs, ys]
            bbox.append(point)
            # print('bx:', bbox)
        boxes_list.append(bbox)
    print(boxes_list)
    return boxes_list

def visualize_polygon(img, img_path, txt_path, output):
    boxes_list = get_boxs_list(txt_path)
    img = draw_bbox(cv2.imread(img_path)[:, :, ::-1], boxes_list)
    name_img = img_path.split('/')[-1]
    if not os.path.exists(output):
        os.mkdir(output)
    output_path_file = os.path.join(output, name_img)
    cv2.imwrite(output_path_file, img)

def isclose(a,b, th):
    return a == b or abs(a-b) < th 


def visualize_line(img_path , final_box, output):
    im = Image.open(img_path) # img1 anh task 1
    draw1 = ImageDraw.Draw(im)
    name_img = img_path.split('/')[-1]
    for box in final_box:
        print('box',box)
        draw1.rectangle(box, outline ="blue") 
    if not os.path.exists(output):
        os.mkdir(output)
    output_img_path = os.path.join(output, name_img)
    im.save(output_img_path,"JPEG")

def get_line(img_path, txt_path, output):
    start_points, end_points,_ = get_boxs(txt_path)

    final_box = []
    temp = []
    temp_x = []
    temp_y = []

    for sp1, ep1 in zip(start_points, end_points):
        list_temp = []
        temp_x = [sp1]
        temp_y = [ep1]
        # temp.append(sp1[0])
        if sp1[1] not in temp:
            for sp2, ep2 in zip(start_points, end_points):
                if isclose(sp1[1],sp2[1],20) and sp2 != sp1:
                    temp_x.append(sp2)
                    temp_y.append(ep2)
                    temp.append(sp2[1])
            # print('temp', temp)
            print(temp_x)
            print('temp_y',temp_y)
            list_temp.append((min(temp_x,key=lambda item:item[0])[0],min(temp_x,key=lambda item:item[1])[1]))
            list_temp.append((max(temp_y,key=lambda item:item[0])[0],max(temp_y,key=lambda item:item[1])[1])) 
            final_box.append(list_temp)
    print(final_box)
    visualize_line(img_path , final_box, output)
    print('len(final_box)',len(final_box))
    return(len(final_box))


def visualize_for_train(input_folder_img, input_folder_img_crop, input_folder_txt,input_GT_csv , output_path):
    output_path = os.path.join(output_path, 'img_visualize_train')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    INPUT_FOLDER_IMAGES = input_folder_img
    INPUT_FOLDER_IMAGES_CROP = input_folder_img_crop
    INPUT_FOLDER_TXT = input_folder_txt
    OUTPUT_VISUALIZE = output_path + '/API'
    OUTPUT_VISUALIZE_BOX_GT = output_path + '/bbox'
    OUTPUT_VISUALIZE_LINE = output_path + '/line'
    OUTPUT_VISUALIZE_MERGE = output_path + '/merge'
    OUTPUT_VISUALIZE_POLYGON = output_path + '/API_polygon'
    path_img, dirs_img, files_img = next(os.walk(INPUT_FOLDER_IMAGES_CROP))
    path_txt, dirs_txt, files_txt = next(os.walk(INPUT_FOLDER_TXT))
    print(len(files_img))
    print(len(files_txt))
    for fn_img in files_img:
        image_path = os.path.join(path_img, fn_img)
        annot_path = os.path.join(path_txt, fn_img.replace('.jpg', '.txt'))
        im = cv2.imread(image_path)
        # if fn_txt.split('.')[0] == fn_img.split('.')[0]:
        print(annot_path, image_path)
        visualize(im,image_path, annot_path, OUTPUT_VISUALIZE)
        visualize_polygon(im, image_path, annot_path, OUTPUT_VISUALIZE_POLYGON)
    
    
    data = pd.read_csv(input_GT_csv)
    path_img, dirs_img, files_img = next(os.walk(INPUT_FOLDER_IMAGES))
    for img_id, anno_polygons in zip(data['img_id'], data['anno_polygons']):
        image_path = os.path.join(path_img, img_id)
        img = cv2.imread(image_path)
        print(anno_polygons)
        anno_polygons_list = ast.literal_eval(anno_polygons)
        for dic in anno_polygons_list:
            print(dic)
            print(dic['bbox'])
            x, y, w, h = dic['bbox']
            xmax = x + w
            ymax = y + h
            cv2.rectangle(img, (x,y), (xmax, ymax), (0,255,0), 2)
        output = OUTPUT_VISUALIZE_BOX_GT
        if not os.path.exists(output):
            os.mkdir(output)
        output_path_file = os.path.join(output, img_id)
        cv2.imwrite(output_path_file, img)
    
    path_img, dirs_img, files_img = next(os.walk(INPUT_FOLDER_IMAGES_CROP))
    path_txt, dirs_txt, files_txt = next(os.walk(INPUT_FOLDER_TXT))
    print(len(files_img))
    print(len(files_txt))
    count_lines = {}
    for fn_img in files_img:
        image_path = os.path.join(path_img, fn_img)
        annot_path = os.path.join(path_txt, fn_img.replace('.jpg', '.txt'))
        im = cv2.imread(image_path)
        # if fn_txt.split('.')[0] == fn_img.split('.')[0]:
        print(annot_path, image_path)
        count_line = get_line(image_path, annot_path, OUTPUT_VISUALIZE_LINE)
        count_lines[fn_img] = count_line
    
    path_img_1, dirs_img_1, files_img_1 = next(os.walk(OUTPUT_VISUALIZE_POLYGON))
    # # path_img_2, dirs_img_2, files_img_2 = next(os.walk(OUTPUT_VISUALIZE))
    for fn_img in files_img_1:
        # try:
        print(fn_img)
        image_path_1 = os.path.join(OUTPUT_VISUALIZE_BOX_GT, fn_img)
        image_path_2 = os.path.join(OUTPUT_VISUALIZE_POLYGON, fn_img)
        image_path_3 = os.path.join(OUTPUT_VISUALIZE, fn_img)
        image_path_4 = os.path.join(OUTPUT_VISUALIZE_LINE, fn_img)
        img1 = cv2.imread(image_path_1) 
        img2 = cv2.imread(image_path_2)
        img3 = cv2.imread(image_path_3)
        img4 = cv2.imread(image_path_4)

        org = (100, 100) 
        fontScale = 2
        font = cv2.FONT_HERSHEY_SIMPLEX 
        color = (0, 255, 0) 
        thickness = 3
        line_count = count_lines[fn_img]
        img1 = cv2.putText(img1, 'DATA BTC', org, font, fontScale, color, thickness, cv2.LINE_AA)
        img2 = cv2.putText(img2, 'DETEC', (img2.shape[1]//2-200, 100), font, fontScale, color, thickness, cv2.LINE_AA)
        img3 = cv2.putText(img3, 'REG', (img2.shape[1]//2-200, 100), font, fontScale, color, thickness, cv2.LINE_AA)
        img4 = cv2.putText(img4, str(line_count) +' LINE', org, font, fontScale, color, thickness, cv2.LINE_AA)

        img1 = cv2.rectangle(img1, (0,0), (img1.shape[1],img1.shape[0]), color, 10)
        img2 = cv2.rectangle(img2, (0,0), (img2.shape[1],img2.shape[0]), color, 10)
        img3 = cv2.rectangle(img3, (0,0), (img3.shape[1],img3.shape[0]), color, 10)
        img4 = cv2.rectangle(img4, (0,0), (img3.shape[1],img3.shape[0]), color, 10)

        image_names = [img1, img2, img3, img4]
        images = []
        total_width = 0 # find the max width of all the images
        max_height = 0 # the total height of the images (vertical stacking)

        for name in image_names:
            # open all images and find their sizes
            images.append(name)
            print(name.shape)
            if images[-1].shape[0] > max_height:
                max_height = images[-1].shape[0]
            total_width += images[-1].shape[1]

        # create a new array with a size large enough to contain all the images
        final_image = np.zeros((max_height,total_width,3),dtype=np.uint8)
        print('final',final_image.shape)

        current_x = 0 # keep track of where your current image was last placed in the y coordinate
        for image in images:
            # add an image to the final array and increment the y coordinate
            print(image.shape)
            print(image.shape[1] + current_x)
            final_image[0:image.shape[0],current_x:image.shape[1]+current_x] = image
            current_x += image.shape[1]

        # cv2.imwrite('fin.PNG',final_image)
        print(fn_img)
        output = OUTPUT_VISUALIZE_MERGE
        if not os.path.exists(output):
            os.mkdir(output)
        output_path_file = os.path.join(output, fn_img)
        cv2.imwrite(output_path_file, final_image)
        # except:
        #     pass
def visualize_for_test(input_folder_img, input_folder_img_crop, input_folder_txt , output_path):
    INPUT_FOLDER_IMAGES = input_folder_img
    INPUT_FOLDER_IMAGES_CROP = input_folder_img_crop
    INPUT_FOLDER_TXT = input_folder_txt
    OUTPUT_VISUALIZE = output_path + '/API'
    OUTPUT_VISUALIZE_BOX_GT = output_path + '/bbox'
    OUTPUT_VISUALIZE_LINE = output_path + '/line'
    OUTPUT_VISUALIZE_MERGE = output_path + '/merge'
    OUTPUT_VISUALIZE_POLYGON = output_path + '/API_polygon'
    path_img, dirs_img, files_img = next(os.walk(INPUT_FOLDER_IMAGES_CROP))
    path_txt, dirs_txt, files_txt = next(os.walk(INPUT_FOLDER_TXT))
    if not os.path.exists(OUTPUT_VISUALIZE):
            os.makedirs(OUTPUT_VISUALIZE)
    if not os.path.exists(OUTPUT_VISUALIZE_POLYGON):
            os.makedirs(OUTPUT_VISUALIZE_POLYGON)
    if not os.path.exists(OUTPUT_VISUALIZE_LINE):
            os.makedirs(OUTPUT_VISUALIZE_LINE)
    print(len(files_img))
    print(len(files_txt))
    for fn_img in files_img:
        image_path = os.path.join(path_img, fn_img)
        # annot_path = os.path.join(path_txt, fn_img.replace('.JPG', '.txt'))
        annot_name = fn_img.split('.')[0] + '.txt'
        annot_path = os.path.join(path_txt, annot_name)
        im = cv2.imread(image_path)
        # if fn_txt.split('.')[0] == fn_img.split('.')[0]:
        print(annot_path, image_path)
        
        visualize(im,image_path, annot_path, OUTPUT_VISUALIZE)
        visualize_polygon(im, image_path, annot_path, OUTPUT_VISUALIZE_POLYGON)

    path_img, dirs_img, files_img = next(os.walk(INPUT_FOLDER_IMAGES_CROP))
    path_txt, dirs_txt, files_txt = next(os.walk(INPUT_FOLDER_TXT))
    print(len(files_img))
    print(len(files_txt))
    count_lines = {}
    for fn_img in files_img:
        image_path = os.path.join(path_img, fn_img)
        # annot_path = os.path.join(path_txt, fn_img.replace('.jpg', '.txt'))
        annot_name = fn_img.split('.')[0] + '.txt'
        annot_path = os.path.join(path_txt, annot_name)
        im = cv2.imread(image_path)
        # if fn_txt.split('.')[0] == fn_img.split('.')[0]:
        print(annot_path, image_path)
        count_line = get_line(image_path, annot_path, OUTPUT_VISUALIZE_LINE)
        count_lines[fn_img] = count_line
    
    path_img_1, dirs_img_1, files_img_1 = next(os.walk(OUTPUT_VISUALIZE_POLYGON))
    # path_img_2, dirs_img_2, files_img_2 = next(os.walk(OUTPUT_VISUALIZE))
    for fn_img in files_img_1:
        print(fn_img)
        image_path_1 = os.path.join(path_img_1, fn_img)
        image_path_2 = os.path.join(OUTPUT_VISUALIZE, fn_img)
        image_path_3 = os.path.join(OUTPUT_VISUALIZE_LINE, fn_img)
        img1 = cv2.imread(image_path_1) 
        img2 = cv2.imread(image_path_2)
        img3 = cv2.imread(image_path_3)

        org = (100, 100) 
        fontScale = 2
        font = cv2.FONT_HERSHEY_SIMPLEX 
        color = (0, 255, 0) 
        thickness = 3
        line_count = count_lines[fn_img]
        img1 = cv2.putText(img1, 'DETEC', org, font, fontScale, color, thickness, cv2.LINE_AA)
        img2 = cv2.putText(img2, 'DETEC & REG', (img2.shape[1]//2-200, 100), font, fontScale, color, thickness, cv2.LINE_AA)
        img3 = cv2.putText(img3, str(line_count) +' LINE', org, font, fontScale, color, thickness, cv2.LINE_AA)

        img1 = cv2.rectangle(img1, (0,0), (img1.shape[1],img1.shape[0]), color, 10)
        img2 = cv2.rectangle(img2, (0,0), (img2.shape[1],img2.shape[0]), color, 10)
        img3 = cv2.rectangle(img3, (0,0), (img3.shape[1],img3.shape[0]), color, 10)

        image_names = [img1, img2, img3]
        images = []
        total_width = 0 # find the max width of all the images
        max_height = 0 # the total height of the images (vertical stacking)

        for name in image_names:
            # open all images and find their sizes
            images.append(name)
            print(name.shape)
            if images[-1].shape[0] > max_height:
                max_height = images[-1].shape[0]
            total_width += images[-1].shape[1]

        # create a new array with a size large enough to contain all the images
        final_image = np.zeros((max_height,total_width,3),dtype=np.uint8)
        print('final',final_image.shape)

        current_x = 0 # keep track of where your current image was last placed in the y coordinate
        for image in images:
            # add an image to the final array and increment the y coordinate
            print(image.shape)
            print(image.shape[1] + current_x)
            final_image[0:image.shape[0],current_x:image.shape[1]+current_x] = image
            current_x += image.shape[1]

        # cv2.imwrite('fin.PNG',final_image)
        print(fn_img)
        output = OUTPUT_VISUALIZE_MERGE
        if not os.path.exists(output):
            os.mkdir(output)
        output_path_file = os.path.join(output, fn_img)
        cv2.imwrite(output_path_file, final_image)

if __name__ == "__main__":
    visualize_for_train(input_folder_img, input_folder_img_crop, input_folder_txt,input_GT_csv , output_path)

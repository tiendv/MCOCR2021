import cv2
import numpy as np
import os
from .submit import get_submit_image, print_output
import os 
import pandas as pd
import random
import json
import time

def add_info_compare(name_tac_gia, input_json_path, input_txt_tac_gia_path, output_path):

    with open(input_txt_tac_gia_path, 'r') as f:
        result_submit_my = f.readlines()
    content_my = [x.strip() for x in result_submit_my]

    for ele in content_my:
        list_info = []
        try:
            img_name, result_value, result_field = ele.split('\t')
        except:
            img_name = ele.strip()
        json_file_name = img_name.replace('.txt', '.json')
        json_visualize_path = os.path.join(input_json_path,json_file_name)
        with open(json_visualize_path, encoding= 'utf-8') as out:
        # keys = out.read().encode('utf-8')
            json_file = json.load(out)
        
        # print(img_id)
        print('ele',ele)
        try:
            img_name, result_value, result_field = ele.split('\t')
            json_file_name = img_name.replace('.txt', '.json')
            json_visualize_path = os.path.join(input_json_path,json_file_name)
            print(json_visualize_path)
            list_field_values = {'TIMESTAMP':''}
            try:
                # with open(json_visualize_path, encoding= 'utf-8') as out:
                #     # keys = out.read().encode('utf-8')
                #     json_file = json.load(out)

                try:
                    field_name_list_my = result_field.split('|||')
                    field_value_list_my = result_value.split('|||')
                    for i, (field_name, field_value) in enumerate(zip(field_name_list_my, field_value_list_my)):
        
                        list_field_values[field_name] += field_value + '|||'
                    print(list_field_values)
                    for field_name, field_value in list_field_values.items():
                        field_dic = {}
                        field_dic['field_name'] = name_tac_gia + '_' + field_name
                        field_dic['field_value'] = field_value[:-3]
                        print(field_dic)
                        list_info.append(field_dic)
                    for i in list_info:
                        json_file['data']['info'].append(i)
                    # json_file['data']['info'] = json_file['data']['info'] list_info
                except:
                    pass
                OUTPUT_FOLDER = os.path.join(output_path, name_tac_gia)
                if not os.path.exists(OUTPUT_FOLDER):
                    os.mkdir(OUTPUT_FOLDER)
                OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, json_file_name)
                with open(OUTPUT_FILE,'w' ,encoding='utf8') as out:
                    print('------------------results.csv 1--------------------------')
                    json.dump(json_file, out, ensure_ascii=False)
            except:
                pass
        except:
            pass
        OUTPUT_FOLDER = os.path.join(output_path, name_tac_gia)
        if not os.path.exists(OUTPUT_FOLDER):
            os.mkdir(OUTPUT_FOLDER)
        OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, json_file_name)
        with open(OUTPUT_FILE,'w' ,encoding='utf8') as out:
            print('------------------results.csv 2--------------------------')
            json.dump(json_file, out, ensure_ascii=False)

def extract_info_1(input_folder_img, input_folder_txt, output_path, json_visualize_path):
    path_img, dirs_img, files_img = next(os.walk(input_folder_img))
    path_txt, dirs_txt, files_txt = next(os.walk(input_folder_txt))
    print(len(files_img))
    # print(len(files_txt))

    result_sub_str = ''
    for fn_img in files_img:
        image_path = os.path.join(path_img, fn_img)
        fn_txt = fn_img.replace('jpg', 'txt')
        annot_path = os.path.join(path_txt, fn_txt)

        print(annot_path, image_path)
        json_file = {"fname": fn_img, "data":{"info":[], "box":[]}}
        list_info = []
        try:

            output_dict = get_submit_image(image_path, annot_path)

            result_value, result_field = print_output(output_dict)
            print(result_value)

            field_name_list_my = result_field.split('|||')
            field_value_list_my = result_value.split('|||')
            list_field_values = {'SELLER':'', 'ADDRESS':'', 'TIMESTAMP':'', 'TOTAL_COST':''}
            for i, (field_name, field_value) in enumerate(zip(field_name_list_my, field_value_list_my)):

                list_field_values[field_name] += field_value + '|||'
            # print(list_field_values)
            for field_name, field_value in list_field_values.items():
                field_dic = {}
                field_dic['field_name'] = field_name + '_my'
                field_dic['field_value'] = field_value[:-3]
                # print('field_dic', field_dic)
                list_info.append(field_dic)
            for i in list_info:
                json_file['data']['info'].append(i)
        

        except Exception as e:
            print(e)
            with open(output_path + 'error.txt', 'w') as f:
                f.write('{}\n'.format(image_path))
                f.write('{}\n'.format(e))
            result_value = 'err'
            result_field = 'err'
        

        json_visualize_file = os.path.join(json_visualize_path, fn_img.replace('jpg', 'json'))
        print(json_visualize_file)
        with open(json_visualize_file, 'w', encoding='utf8') as output_file:
            json.dump(json_file, output_file, ensure_ascii=False)
            print('output_path: ',json_visualize_file, '----ok----')  


        result_sub_str += fn_img + '\t' + result_value + '\t' + result_field + '\n'
    output_file_txt = os.path.join(output_path, 'results.txt')
    with open(output_file_txt, 'w') as out:
        out.write(result_sub_str)   

    with open(output_file_txt, 'r') as f:
        result_submit_txt = f.readlines()

    content = [x.strip() for x in result_submit_txt]
    result = pd.DataFrame(columns = ['img_id', 'anno_image_quality','anno_texts'])

    for ele in content:
        try:
            img_name, result_value, result_field = ele.split('\t')
        except:
            img_name = ele
            result_value = ''
        print(result_field)
        # anno_image_quality = round(int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1),5)
        field_name_list_my = result_field.split('|||')
        field_value_list_my = result_value.split('|||')
        list_field_values = {'SELLER':'', 'ADDRESS':'', 'TIMESTAMP':'', 'TOTAL_COST':''}
        # field_value_new = ''
        # for i, (field_name, field_value) in enumerate(zip(field_name_list_my, field_value_list_my)):
        #     if field_name != 'ADDRESS':
        #         print(field_name)
        #         field_value_new += field_value + '|||'
        #     else:
        #         print('------------------addd---------------------')
        #         field_value_new += ' |||'
        anno_image_quality = random.uniform(0.5, 1)
        # print('field_value_new',field_value_new)
        # print('result_field',result_field)
        result.loc[len(result)] = [img_name,anno_image_quality,result_value]
        # print(result.loc[len(result)])
    print('------------------results.csv--------------------------')
    result.to_csv(output_path + '/results.csv',index=False)
    return output_path + '/results.txt'

def extract_info(input_folder_img, input_folder_txt, output_path, json_visualize_path):
    t1 = time.time()
    print('extract_info')
    path_img, dirs_img, files_img = next(os.walk(input_folder_img))
    path_txt, dirs_txt, files_txt = next(os.walk(input_folder_txt))
    print(len(files_img))
    # print(len(files_txt))

    result_sub_str = ''
    for fn_img in files_img:
        image_path = os.path.join(path_img, fn_img)
        fn_txt = fn_img.replace('jpg', 'txt')
        annot_path = os.path.join(path_txt, fn_txt)

        print(annot_path, image_path)
        json_file = {"fname": fn_img, "data":{"info":[], "box":[]}}
        list_info = []
        try:

            output_dict = get_submit_image(image_path, annot_path)

            result_value, result_field = print_output(output_dict)
            print(result_value)

            field_name_list_my = result_field.split('|||')
            field_value_list_my = result_value.split('|||')
            list_field_values = {'SELLER':'', 'ADDRESS':'', 'TIMESTAMP':'', 'TOTAL_COST':''}
            for i, (field_name, field_value) in enumerate(zip(field_name_list_my, field_value_list_my)):

                list_field_values[field_name] += field_value + '|||'
            # print(list_field_values)
            for field_name, field_value in list_field_values.items():
                field_dic = {}
                field_dic['field_name'] = field_name + '_my'
                field_dic['field_value'] = field_value[:-3]
                # print('field_dic', field_dic)
                list_info.append(field_dic)
            for i in list_info:
                json_file['data']['info'].append(i)
        

        except Exception as e:
            print(e)
            with open(output_path + 'error.txt', 'w') as f:
                f.write('{}\n'.format(image_path))
                f.write('{}\n'.format(e))
            result_value = '|||'
            result_field = '|||'
        

        json_visualize_file = os.path.join(json_visualize_path, fn_img.replace('jpg', 'json'))
        print(json_visualize_file)
        with open(json_visualize_file, 'w', encoding='utf8') as output_file:
            json.dump(json_file, output_file, ensure_ascii=False)
            print('output_path: ',json_visualize_file, '----ok----')  


        result_sub_str += fn_img + '\t' + result_value + '\t' + result_field + '\n'
    output_file_txt = os.path.join(output_path, 'results.txt')
    with open(output_file_txt, 'w') as out:
        out.write(result_sub_str)   
    t2 = time.time() - t1 
    with open(output_file_txt, 'r') as f:
        result_submit_txt = f.readlines()

    content = [x.strip() for x in result_submit_txt]
    result = pd.DataFrame(columns = ['img_id', 'anno_image_quality','anno_texts'])

    for ele in content:
        try:
            img_name, result_value, result_field = ele.split('\t')
        except:
            img_name = ele
            result_value = ''
        # print(result_field)
        # anno_image_quality = round(int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1),5)
        field_name_list_my = result_field.split('|||')
        field_value_list_my = result_value.split('|||')
        list_field_values = {'SELLER':'', 'ADDRESS':'', 'TIMESTAMP':'', 'TOTAL_COST':''}
        field_value_new = ''
        field_name_list_swap = []
        field_value_list_swap = []
        for i, (field_name, field_value) in enumerate(zip(field_name_list_my, field_value_list_my)):
            if field_name == 'SELLER':
                field_name_list_swap.append(field_name)
                field_value_list_swap.append(field_value)
        for i, (field_name, field_value) in enumerate(zip(field_name_list_my, field_value_list_my)):
            if field_name == 'ADDRESS':
                field_name_list_swap.append(field_name)
                field_value_list_swap.append(field_value)
        for i, (field_name, field_value) in enumerate(zip(field_name_list_my, field_value_list_my)):
            if field_name == 'TIMESTAMP':
                field_name_list_swap.append(field_name)
                field_value_list_swap.append(field_value)
        for i, (field_name, field_value) in enumerate(zip(field_name_list_my, field_value_list_my)):
            if field_name == 'TOTAL_COST':
                field_name_list_swap.append(field_name)
                field_value_list_swap.append(field_value)

        submit_full = True
        if submit_full:
            for i, (field_name, field_value) in enumerate(zip(field_name_list_swap, field_value_list_swap)):
                field_value_new += field_value + '|||'
        else:
            for i, (field_name, field_value) in enumerate(zip(field_name_list_swap, field_value_list_swap)):
                if field_name != 'TOTAL_COST' and field_name != 'TOTAL_COST' and field_name != 'TOTAL_COST':
                    field_value_new += ''
                else: 
                    field_value_new += field_value + '|||'
        anno_image_quality = random.uniform(0.5, 1)
        print('field_value_new',field_value_new)
        # print('result_field',result_field)
        if field_value_new == '' or field_value_new == ' ':
            field_value_new = '|||'
        elif field_value_new[-3:] == '|||':
            field_value_new = field_value_new[:-3]
        result.loc[len(result)] = [img_name,anno_image_quality,field_value_new]
        # print(result.loc[len(result)])
    
    print('------------------results.csv--------------------------')
    print('-----------', t2)
    result.to_csv(output_path + '/results.csv',index=False)
    return output_path + '/results.txt'
    
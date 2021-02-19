import os 
import pandas as pd 
import shutil 
import json
import ast


OUTPUT_FOLDER = 'visualize_train_json_crop'
if not os.path.exists(OUTPUT_FOLDER):
    os.mkdir(OUTPUT_FOLDER)



with open('/home/huy/TEST_SUBMIT/MC-OCR-RIVF2020/mc-ocr_rivf2020/post_processing/result_submit_train_crop.txt', 'r') as f:
    result_submit_my = f.readlines()

content_my = [x.strip() for x in result_submit_my]

    # print(anno_image_quality)
for ele in content_my:
    
    

    result_txt_list = {"fname": '', "data":{"info":[], "box":[]}}
    list_info = []
    
    # print(img_id)
    img_name, result_value, result_field = ele.split('\t')
    json_file_name = img_name.replace('.jpg', '.json')
    json_visualize_path = os.path.join(OUTPUT_FOLDER,json_file_name)
    print(json_visualize_path)
    list_field_values = {'SELLER':'', 'ADDRESS':'', 'TIMESTAMP':'', 'TOTAL_COST':''}
    try:
        with open(json_visualize_path, encoding= 'utf-8') as out:
            # keys = out.read().encode('utf-8')
            json_file = json.load(out)

        try:
            field_name_list_my = result_field.split('|||')
            field_value_list_my = result_value.split('|||')
            for i, (field_name, field_value) in enumerate(zip(field_name_list_my, field_value_list_my)):
 
                list_field_values[field_name] += field_value + '|||'
            print(list_field_values)
            for field_name, field_value in list_field_values.items():
                field_dic = {}
                field_dic['field_name'] = field_name + '_my'
                field_dic['field_value'] = field_value[:-3]
                print(field_dic)
                list_info.append(field_dic)
            for i in list_info:
                json_file['data']['info'].append(i)
            # json_file['data']['info'] = json_file['data']['info'] list_info
        except:
            pass
        with open(json_visualize_path,'w' ,encoding='utf8') as out:
            json.dump(json_file, out, ensure_ascii=False)
    except:
        pass

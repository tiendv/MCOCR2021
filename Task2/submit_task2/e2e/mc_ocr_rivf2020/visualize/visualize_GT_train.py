import os 
import pandas as pd 
import shutil 
import json
import ast

data = pd.read_csv("/home/huy/TEST_SUBMIT/MC-OCR-RIVF2020/mc-ocr_rivf2020/mcocr_public_train_test_shared_data/mcocr_train_data/mcocr_train_df.csv")

OUTPUT_FOLDER = 'visualize_train_json_crop'
if not os.path.exists(OUTPUT_FOLDER):
    os.mkdir(OUTPUT_FOLDER)

for img_id, anno_polygons, anno_num, anno_texts, anno_labels, anno_image_quality in zip(data['img_id'], data['anno_polygons'], data['anno_num'], data['anno_texts'], data['anno_labels'], data['anno_image_quality']):
    # print(anno_image_quality)
    result_txt_list = {"fname": '', "data":{"info":[], "box":[]}}
    list_info = []
    print(img_id)
    list_field_values = {'SELLER':'', 'ADDRESS':'', 'TIMESTAMP':'', 'TOTAL_COST':''}
    try:
        field_name_list = anno_labels.split('|||')
        field_value_list = anno_texts.split('|||')
        for j, (field_name, field_value) in enumerate(zip(field_name_list, field_value_list)):
            
            print(field_name)
            list_field_values[field_name] += field_value + '|||'
        print(list_field_values)
        for field_name, field_value in list_field_values.items():
            field_dic = {}
            field_dic['field_name'] = field_name
            field_dic['field_value'] = field_value[:-3]
            print(field_dic)
            list_info.append(field_dic)
        field_dic = {}
        field_dic['field_name'] = 'SCORE'
        field_dic['field_value'] = anno_image_quality
        list_info.append(field_dic)
    except:
        pass
    print(list_info)
    result_txt_list['fname'] = img_id
    result_txt_list['data']['info'] = list_info
    # anno_polygons_list = json.loads(anno_polygons)
    anno_polygons_list = ast.literal_eval(anno_polygons)
    result_txt_list['data']['box'] = anno_polygons_list

    output_json_name = img_id.replace('.jpg', '.json')
    output_path = os.path.join(OUTPUT_FOLDER,output_json_name)

    with open(output_path, 'w', encoding='utf8') as output_file:
        json.dump(result_txt_list, output_file, ensure_ascii=False)
        print('output_path: ',output_path, '----ok----')

import os 
import pandas as pd 
import shutil 
import json
import ast

# def output_json_for_test(input_file_result_txt ,output_path):
#     with open(input_file_result_txt, 'r') as f:
#     result_submit_my = f.readlines()

#     content_my = [x.strip() for x in result_submit_my]
#     print(content_my)

#         # print(anno_image_quality)
#     for ele in content_my:
#         result_txt_list = {"fname": '', "data":{"info":[], "box":[]}}
#         list_info = []
        
#         # print(img_id)
#         img_name, result_value, result_field = ele.split('\t')
#         json_file_name = img_name.replace('.jpg', '.json')
#         try:
#             field_name_list_my = result_field.split('|||')
#             field_value_list_my = result_value.split('|||')
#             list_field_values = {'SELLER':'', 'ADDRESS':'', 'TIMESTAMP':'', 'TOTAL_COST':''}
#             for i, (field_name, field_value) in enumerate(zip(field_name_list_my, field_value_list_my)):

#                 list_field_values[field_name] += field_value + '|||'
#             # print(list_field_values)
#             for field_name, field_value in list_field_values.items():
#                 field_dic = {}
#                 field_dic['field_name'] = field_name + '_my'
#                 field_dic['field_value'] = field_value[:-3]
#                 # print('field_dic', field_dic)
#                 list_info.append(field_dic)
#             for i in list_info:
#                 json_file['data']['info'].append(i)
#         except Exception as e:
#             print(e)
#             with open(output_path + 'error.txt', 'w') as f:
#                 f.write('{}\n'.format(image_path))
#                 f.write('{}\n'.format(e))
#             result_value = 'Error'
#             result_field = 'Error'
#         json_visualize_file = os.path.join(output_path, json_file_name)
#         print(json_visualize_file)
#         with open(json_visualize_file, 'w', encoding='utf8') as output_file:
#             json.dump(json_file, output_file, ensure_ascii=False)
#             print('output_path: ',json_visualize_file, '----ok----') 

# def output_json_for_test_csv(input_file_result_csv ,output_path):
#     data = pd.read_csv(input_file_result_csv)
#     with open(input_file_result_txt, 'r') as f:
#     result_submit_my = f.readlines()

#     content_my = [x.strip() for x in result_submit_my]
#     print(content_my)

#         # print(anno_image_quality)
#     for ele in content_my:
#         result_txt_list = {"fname": '', "data":{"info":[], "box":[]}}
#         list_info = []
        
#         # print(img_id)
#         img_name, result_value, result_field = ele.split('\t')
#         json_file_name = img_name.replace('.jpg', '.json')
#         try:
#             field_name_list_my = result_field.split('|||')
#             field_value_list_my = result_value.split('|||')
#             list_field_values = {'SELLER':'', 'ADDRESS':'', 'TIMESTAMP':'', 'TOTAL_COST':''}
#             for i, (field_name, field_value) in enumerate(zip(field_name_list_my, field_value_list_my)):

#                 list_field_values[field_name] += field_value + '|||'
#             # print(list_field_values)
#             for field_name, field_value in list_field_values.items():
#                 field_dic = {}
#                 field_dic['field_name'] = field_name + '_my'
#                 field_dic['field_value'] = field_value[:-3]
#                 # print('field_dic', field_dic)
#                 list_info.append(field_dic)
#             for i in list_info:
#                 json_file['data']['info'].append(i)
#         except Exception as e:
#             print(e)
#             with open(output_path + 'error.txt', 'w') as f:
#                 f.write('{}\n'.format(image_path))
#                 f.write('{}\n'.format(e))
#             result_value = 'Error'
#             result_field = 'Error'
#         json_visualize_file = os.path.join(output_path, json_file_name)
#         print(json_visualize_file)
#         with open(json_visualize_file, 'w', encoding='utf8') as output_file:
#             json.dump(json_file, output_file, ensure_ascii=False)
#             print('output_path: ',json_visualize_file, '----ok----') 


def output_json_for_train(data_BTC_csv,result_train ,output_folder):
    data = pd.read_csv(data_BTC_csv)

    OUTPUT_FOLDER = output_folder
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

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
    
    
    with open(result_train, 'r') as f:
        result_submit_my = f.readlines()

    content_my = [x.strip() for x in result_submit_my]
    print(content_my)

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

import os
import shutil

INPUT_FOLDER_1 = 'raw_data_img/test_29_09_38_rotate90_update_dic_weight_balance'
INPUT_FOLDER_2 = 'output_pipline/detec_reg/test_29_09_38_rotate90_update_dic_weight_balance'
OUTPUT_FOLDER = 'predict_lai'

if __name__ == "__main__":
    path_1, dirs_1, files_1 = next(os.walk(INPUT_FOLDER_1))
    path_2, dirs_2, files_2 = next(os.walk(INPUT_FOLDER_2))
    print(len(files_1))
    print(len(files_2))
    for f1 in files_1:
        img_path = os.path.join(INPUT_FOLDER_1, f1)
        txt_path = os.path.join(INPUT_FOLDER_2, f1.replace('jpg', 'txt'))
        output = os.path.join(OUTPUT_FOLDER, f1)
        if not os.path.exists(OUTPUT_FOLDER):
            os.mkdir(OUTPUT_FOLDER)

        if os.path.exists(txt_path):
            # print('ok')
            pass
        else:
            print(f1)
            shutil.copyfile(img_path, output)

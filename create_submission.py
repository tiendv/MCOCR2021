import os
import csv
import cv2
import argparse
import numpy as np
from keras import models
from tqdm import tqdm
from cropper import crop_receipt

def create_submission(input_size, model, test_img_folder, sample_csv_path, result_path):
	quality_list = []
	with open(sample_csv_path, "r", encoding="utf8") as fi:
		csv_reader = csv.DictReader(fi)
		csv_reader = list(csv_reader)

		csv_reader = [dict(d) for d in csv_reader]
		model = models.load_model(model)

		for row in tqdm(csv_reader):
			img_id = row["img_id"]
			image = cv2.imread(os.path.join(test_img_folder, img_id))
			image = crop_receipt(image)
			image = cv2.resize(image, (input_size, input_size))
			image = image / 255.0

			image = image.reshape((1,input_size,input_size,3))
			
			quality = model.predict(image)[0][0]
			quality_list.append(str(quality))

	with open(result_path, "w") as fo:
		fo.write("\n".join(quality_list))

if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("-t", "--test-img-folder", type=str, required=True,
		help="Path to input test images folder")
	ap.add_argument("-c", "--sample-csv-path", type=str, required=True,
		help="Path to sample csv of private test")
	ap.add_argument("-m", "--model-path", type=str, required=True,
		help="Path to the saved model")
	ap.add_argument("-s", "--input-size", type=int, required=True,
		help="Input shape", default=128)
	ap.add_argument("-r", "--result-path", type=str, required=True,
		help="Result path")
	args = vars(ap.parse_args())

	#"./data/private_test/mcocr_private_test_data/mcocr_test_samples_df.csv"
	create_submission(args["input_size"], args["model_path"], args["test_img_folder"], args["sample_csv_path"], args["result_path"])
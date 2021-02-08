import os
import csv
import cv2
import argparse
import numpy as np
from keras import models
from tqdm import tqdm
from cropper import crop_receipt

def create_submission(model, test_img_folder, sample_csv_path, result_path):
	img_id_list = []
	quality_list = []

	# Read sample submission csv file
	with open(sample_csv_path, "r", encoding="utf8") as fi:
		csv_reader = csv.DictReader(fi)
		csv_reader = list(csv_reader)
		csv_reader = [dict(d) for d in csv_reader]
		model = models.load_model(model)

		input_size = model.layers[0].input_shape[1]

		for row in tqdm(csv_reader):
			img_id = row["img_id"]
			img_id_list.append(img_id)

			image = cv2.imread(os.path.join(test_img_folder, img_id))
			image = crop_receipt(image)
			image = cv2.resize(image, (input_size, input_size))
			image = image / 255.0

			image = image.reshape((1,input_size,input_size,3))
			
			quality = model.predict(image)[0][0]
			quality_list.append(str(quality))

	# Write submission csv file
	with open(result_path, "w") as fo:
		header = ["img_id", "anno_image_quality"]
		writer = csv.DictWriter(fo, fieldnames=header)

		writer.writeheader()
		for i, img_id in enumerate(img_id_list):
			writer.writerow({"img_id": img_id_list[i], "anno_image_quality": quality_list[i]})

if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("-t", "--test-img-folder", type=str, required=True,
		help="Path to input test images folder")
	ap.add_argument("-c", "--sample-csv-path", type=str, required=True,
		help="Path to sample submission csv in private test")
	ap.add_argument("-m", "--model-path", type=str, required=True,
		help="Path to the saved model")
	ap.add_argument("-r", "--result-path", type=str, required=True,
		help="The destination path to save submission csv file")
	args = vars(ap.parse_args())

	create_submission(args["model_path"], args["test_img_folder"], args["sample_csv_path"], args["result_path"])
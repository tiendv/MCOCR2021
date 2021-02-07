import os
import csv
import cv2
import numpy as np
import time
from tqdm import tqdm
from cropper import crop_receipt


def load_data(images_folder_path, csv_path, input_size):
	"""Load training data

	Args:\\
		images_folder_path ([str]): Path to the training images folder \\
		csv_path ([str]): Path to the training csv \\
		input_size ([int]): The size that all images will be resized to \\

	Returns:
		images_list ([np.array]): Array of training images, each image is an numpy array \\
		image_quality_list ([np.array]): Array of image's qualitys a.k.a labels, each element is a float
	"""	
	with open(csv_path, "r", encoding='utf8') as fi:
		csv_reader = csv.DictReader(fi)
		csv_reader = list(csv_reader)

		csv_reader = [dict(d) for d in csv_reader]

		images_list = []
		image_quality_list = []
		for row in tqdm(csv_reader):
			img_id = row["img_id"]
			img_quality = row["anno_image_quality"]
			
			img_path = os.path.join(images_folder_path, img_id)
			if os.path.isfile(img_path):
				image = cv2.imread(img_path)
				image = crop_receipt(image)
				image = cv2.resize(image, (input_size, input_size))
				
				images_list.append(image / 255.0)
				image_quality_list.append(float(img_quality))
	
	image_quality_list = np.array(image_quality_list)

	return np.array(images_list), image_quality_list

if __name__ == "__main__":
	load_data("./data/train/images", "./data/train/mcocr_train_df.csv", 64)

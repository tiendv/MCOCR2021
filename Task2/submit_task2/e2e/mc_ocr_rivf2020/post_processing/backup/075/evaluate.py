import argparse
import logging
import sys
import csv
import numpy as np

logging.basicConfig(
	format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')

def create_groundtruth_list(field : str) -> list:
	groundtruth_list = []
	with open("train_df/mcocr_train_df.txt", "r", encoding="utf-8") as fi:
		csv_reader = csv.DictReader(fi)
		csv_reader = list(csv_reader)
		
		csv_reader = [dict(d) for d in csv_reader]
		csv_reader = sorted(csv_reader, key=lambda x: x['img_id'])
		for row in csv_reader:
			if field == "FULL":
				groundtruth_list.append(row["anno_texts"])
			else:
				text_list = row["anno_texts"].split("|||")
				label_list = row["anno_labels"].split("|||")
				try:
					field_indexs = [i for i in range(len(label_list)) if label_list[i] == field]
					text_list = np.array(text_list)
					text_str = "|||".join(text_list[field_indexs])
				except:
					text_str = ""
				
				groundtruth_list.append(text_str)

	return groundtruth_list

def create_result_list(results_txt_path, field) -> list:
	result_list = []
	
	with open(results_txt_path, "r", encoding="utf-8") as fi:
		contents = fi.readlines()
		contents = [item.split("\t") for item in contents]
		contents = sorted(contents)
	
	for item in contents:
		if field == "FULL":
			result_list.append(item[1])
		else:
			text_list = item[1].split("|||")
			label_list = item[2].split("|||")
			try:
				field_indexs = [i for i in range(len(label_list)) if label_list[i] == field]
				text_list = np.array(text_list)
				text_str = "|||".join(text_list[field_indexs])
			except:
				text_str = ""
			
			result_list.append(text_str)

	return result_list

def levenshtein(u, v):
	prev = None
	curr = [0 for i in range(1, len(v) + 2)]
	# Operations: (SUB, DEL, INS)
	prev_ops = None
	curr_ops = [(0, 0, i) for i in range(len(v) + 1)]
	for x in range(1, len(u) + 1):
		prev, curr = curr, [x] + ([None] * len(v))
		prev_ops, curr_ops = curr_ops, [(0, x, 0)] + ([None] * len(v))
		for y in range(1, len(v) + 1):
			delcost = prev[y] + 1
			addcost = curr[y - 1] + 1
			subcost = prev[y - 1] + int(u[x - 1] != v[y - 1])
			curr[y] = min(subcost, delcost, addcost)
			if curr[y] == subcost:
				(n_s, n_d, n_i) = prev_ops[y - 1]
				curr_ops[y] = (n_s + int(u[x - 1] != v[y - 1]), n_d, n_i)
			elif curr[y] == delcost:
				(n_s, n_d, n_i) = prev_ops[y]
				curr_ops[y] = (n_s, n_d + 1, n_i)
			else:
				(n_s, n_d, n_i) = curr_ops[y - 1]
				curr_ops[y] = (n_s, n_d, n_i + 1)
	return curr[len(v)], curr_ops[len(v)]

def evaluate(results_txt_path : str, field: str) -> float:
	"""
	Evaluate the result file in a specific field or full field 

	Args: 
		results_txt_path (str): The path of result file \n
		field (str): The name of field you want to evaluate

	Returns:\n
		float: CER
	"""	

	field_list = ["FULL", "ADDRESS", "SELLER", "TIMESTAMP", "TOTAL_COST"]

	field = field.strip().upper()

	if field not in field_list:
		logging.error(f"The field name '{field}' does not exist")
		return -1

	ref = create_groundtruth_list(field)
	hyp = create_result_list(results_txt_path, field)

	if len(ref) != len(hyp):
			logging.error(
				'The number of reference and transcription sentences does not '
				'match (%d vs. %d)', len(ref), len(hyp))
			return -1
	wer_s, wer_i, wer_d, wer_n = 0, 0, 0, 0
	cer_s, cer_i, cer_d, cer_n = 0, 0, 0, 0
	sen_err = 0
	for n in range(len(ref)):
		# update CER statistics
		_, (s, i, d) = levenshtein(ref[n], hyp[n])
		cer_s += s
		cer_i += i
		cer_d += d
		cer_n += len(ref[n])
		# update WER statistics
		_, (s, i, d) = levenshtein(ref[n].split(), hyp[n].split())
		wer_s += s
		wer_i += i
		wer_d += d
		wer_n += len(ref[n].split())
		# update SER statistics
		if s + i + d > 0:
			sen_err += 1

	cer = (cer_s + cer_i + cer_d) / cer_n

	return cer
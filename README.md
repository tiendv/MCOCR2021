## MC-OCR 2021 - Receipt Image Quality Evaluation using EfficientNet

This repository contains our source code of Task1 in the RIVF2021 MC-OCR Competition.

### Introduction

The challenge Task 1 of MC-OCR 2021 required participating teams to submit receipt image quality. Receipt image quality is  measured  by  the  ratio  of text lines associated with the “clear” label evaluated by human  annotators.  The  quality  ranges  from  0  to  1  in  which, score of 1 means the highest quality and score of 0 means the lowest quality.

Detailed information of NVIDIA AICity Challenge 2019 can be found [here](https://rivf2021-mc-ocr.vietnlp.com/).
![overview](overview.png)

### Requirements
- Python 3.6
- OpenCV
- sklearn
- tensorflow
- Keras

### Installation

`pip3 install -r requirements.txt`

### Train EfficientNet model

To adjust the hyperparameter of training model, please access `config.py`

To train model, run the command below:
`python3 train.py`

### Create submission based on saved model

`python3 create_submission.py -t <test_img_folder> -c <sample_csv_path> -m <saved_model_path> -r <result_path>`

- `test_img_folder`: Path to input test images folder.
- `sample_csv_path`: Path to sample submission csv in private test.
- `saved_model_path`: Path to the saved model.
- `result_path`: The destination path to save submission csv file.

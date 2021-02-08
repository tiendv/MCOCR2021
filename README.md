## MC-OCR 2021 - Receipt Image Quality Evaluation using EfficientNet

This repository contains our source code of Task1 in the RIVF2021 MC-OCR Competition.

### Introduction

The challenge Task 1 of MC-OCR 2021 required participating teams to submit receipt image quality. Receipt image quality is  measured  by  the  ratio  of text lines associated with the “clear” label evaluated by human  annotators.  The  quality  ranges  from  0  to  1  in  which, score of 1 means the highest quality and score of 0 means the lowest quality.
Detailed information of NVIDIA AICity Challenge 2019 can be found [here](https://rivf2021-mc-ocr.vietnlp.com/).

![overview](overview.jpg)

### Requirements
- Python 3.6
- OpenCV
- sklearn
- tensorflow
- Keras

### Installation

`pip3 install -r requirements.txt`

### Train EfficientNet model

`python3 train.py`

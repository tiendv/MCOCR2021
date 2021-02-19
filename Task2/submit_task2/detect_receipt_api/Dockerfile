FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

RUN apt update

COPY . /api_detec_receipt_corner

RUN apt install -y libgl1 libglib2.0-0 libxrender1 libsm6

RUN cd /api_detec_receipt_corner && \
    pip install -e detectron2 && \
    pip install -r requirements.txt

WORKDIR /api_detec_receipt_corner

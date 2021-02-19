FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

RUN apt update

COPY . /task2

RUN apt install -y libgl1 libglib2.0-0 libxrender1 libsm6

RUN cd /task2 && \
    pip install -r requirements.txt

WORKDIR /task2

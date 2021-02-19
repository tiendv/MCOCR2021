FROM ubuntu:18.04

RUN apt update

RUN apt-get install -y git \
    software-properties-common \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get install -y locales locales-all
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt install python3.6 -y && \
    apt install python3-distutils -y && \
    apt install python3.6-dev -y && \
    apt install build-essential -y && \
    apt-get install python3-pip -y && \
    apt update && apt install -y libsm6 libxext6 && \
    apt-get install -y libxrender-dev

RUN apt update

COPY . /backup

RUN cd backup && \
    apt install apturl -y && \
    pip3 install easydict imutils pandas Pillow matplotlib numpy scipy requests Cython && \
    pip3 install --upgrade && \
    pip3 install scikit-learn && \
    pip3 install --upgrade && \
    pip3 install --upgrade pip && \
    pip3 install tqdm recognizers_text recognizers_date_time pyyaml lmdb && \
    pip3 install -r requirements.txt
    #pip3 install opencv-contrib-python && \ 
    #pip3 install opencv-python

RUN pip3 install opencv-python -vvv

# RUN pip3 install -r requirements.txt

#RUN software-properties-common

WORKDIR /backup

# opencv-python, pyyaml



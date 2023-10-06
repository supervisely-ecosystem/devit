FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install python3-pip -y
RUN apt-get install -y git
RUN python3 -m pip install --upgrade pip

RUN git clone https://github.com/mlzxy/devit.git
RUN pip3 install -r devit/requirements.txt
RUN pip3 install -e ./devit

RUN python3 -m pip install supervisely==6.72.145

RUN apt-get update
RUN apt-get install ffmpeg libgeos-dev libsm6 libxext6 libexiv2-dev libxrender-dev libboost-all-dev -y
RUN pip install opencv-python

RUN apt-get -y install wget
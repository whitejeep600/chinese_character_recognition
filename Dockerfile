FROM debian:buster
MAINTAINER antoni_maciag

RUN apt-get -y update
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN apt-get -y install libgl1-mesa-glx
RUN pip3 install --upgrade pip
RUN pip3 install torch
RUN pip3 install OpenCV-contrib-python
RUN pip3 install pillow
RUN pip3 install torchvision
RUN pip3 install tqdm

ADD all_characters.json all_characters.json
ADD model.py model.py
ADD setup.sh setup.sh
ADD preprocessing.py preprocessing.py 
ADD infer.py infer.py
ADD utils.py utils.py
ADD constants.py constants.py
ADD readers.py readers.py
ADD label_to_int.json label_to_int.json
ADD dataset.py dataset.py 
ADD checkpoint/ checkpoint/

FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel

# Root user permissions
USER root

ENV TZ=America/Santiago
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y tzdata

# Update
RUN apt-get update && apt-get install -y build-essential

RUN apt-get install git-all -y

# Install mujoco
RUN apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3
RUN git clone https://github.com/openai/mujoco-py
RUN pip install -e ./mujoco-py

# Install gym and dependencies
RUN apt-get install -y libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb xserver-xephyr ffmpeg curl patchelf libglfw3 libglfw3-dev cmake zlib1g zlib1g-dev swig
RUN pip install gym[box2d]==0.21.0 pyvirtualdisplay > /dev/null 2>&1

RUN apt-get update && apt install -y python3-tk

# Install requirements
COPY requirements.txt /opt/app/requirements.txt
RUN pip install -r /opt/app/requirements.txt

# RUN pip install --upgrade git+https://github.com/VincentStimper/normalizing-flows.git

# Copy code, uncomment this before build image
WORKDIR /home/user/workspace
#COPY . .
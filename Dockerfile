# Base image
FROM anibali/pytorch:1.5.0-cuda10.2
# Root user permissions
USER root

# Update
RUN apt-get update && apt-get install -y build-essential

# Install mujoco
RUN sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3
RUN git clone https://github.com/openai/mujoco-py
RUN pip install -e ./mujoco-py

# Install gym and dependencies
RUN apt-get install -y libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb ffmpeg curl patchelf libglfw3 libglfw3-dev cmake zlib1g zlib1g-dev swig
RUN pip install gym[box2d]==0.17.2 pyvirtualdisplay > /dev/null 2>&1

# Install requirements
COPY requirements.txt /opt/app/requirements.txt
RUN pip install -r /opt/app/requirements.txt
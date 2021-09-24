FROM nvidia/cudagl:11.1.1-devel-ubuntu18.04

ARG USER_NAME
ARG USER_PASSWORD
ARG USER_ID
ARG USER_GID

RUN apt-get update
RUN apt install sudo
RUN useradd -ms /bin/bash $USER_NAME
RUN usermod -aG sudo $USER_NAME
RUN yes $USER_PASSWORD | passwd $USER_NAME

# set uid and gid to match those outside the container
RUN usermod -u $USER_ID $USER_NAME
RUN groupmod -g $USER_GID $USER_NAME

# work directory
WORKDIR /home/$USER_NAME

# install system dependencies
COPY ./scripts/install_deps.sh /tmp/install_deps.sh
RUN yes "Y" | /tmp/install_deps.sh

# setup python environment
RUN cd $WORKDIR

# install python requirements
# RUN sudo python3 -m pip install --upgrade pip && \ 
#     sudo python3 -m pip install --upgrade

# install pip3
RUN apt-get -y install python3-pip
RUN sudo python3 -m pip install --upgrade pip

# install pytorch
RUN sudo pip3 install \
   torch==1.9.1+cu111 \
   torchvision==0.10.1+cu111 \
   -f https://download.pytorch.org/whl/torch_stable.html

# install GLX-Gears (for debugging)
RUN apt-get update && apt-get install -y \
   mesa-utils \
   python3-setuptools \
   && rm -rf /var/lib/apt/lists/*


RUN sudo pip3 install \
   absl-py>=0.7.0  \
   gym==0.17.3 \
   pybullet>=3.0.4 \
   matplotlib>=3.1.1 \
   opencv-python>=4.1.2.30 \
   meshcat>=0.0.18 \
   scipy==1.4.1 \
   scikit-image==0.17.2 \
   transforms3d==0.3.1 \
   pytorch_lightning==1.0.3 \
   tdqm \
   hydra-core==1.0.5 \
   wandb \
   transformers==4.3.2 \
   kornia \
   ftfy \
   regex \
   ffmpeg \
   imageio-ffmpeg


# change ownership of everything to our user
RUN mkdir /home/$USER_NAME/cliport
RUN cd /home/$USER_NAME/cliport && echo $(pwd) && chown $USER_NAME:$USER_NAME -R .
RUN echo "export CLIPORT_ROOT=~/cliport" >> /home/$USER_NAME/.bashrc
RUN echo "export PYTHONPATH=$PYTHONPATH:~/cliport" >> /home/$USER_NAME/.bashrc

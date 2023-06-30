#!/bin/bash

set -euxo pipefail

# update
apt-get update

# common
apt-get -y install software-properties-common

# python source list
add-apt-repository -y ppa:deadsnakes/ppa

# dependencies
apt-get update
DEBIAN_FRONTEND=noninteractive apt install --no-install-recommends \
  curl \
  terminator \
  tmux \
  vim \
  gedit \
  git \
  openssh-client \
  openssh-server \
  unzip \
  htop \
  apt-utils \
  usbutils \
  dialog \
  python3.8-venv \
  python3.8-dev \
  ffmpeg \
  nvidia-settings \
  libffi-dev \
  libfreetype6-dev \
  libgl1-mesa-dev \
  flex \
  bison \
  build-essential \
  gcc \
  git \
  wget \
  module-init-tools \
  pciutils \
  xserver-xorg \
  xserver-xorg-video-fbdev \
  xauth \
  python3-pip \
  python3-ipdb \
  python3-tk 
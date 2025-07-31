# Use an NVIDIA CUDA image as the base
FROM nvidia/cuda:12.1.0-devel-ubuntu20.04

# Set up environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="${PATH}:/home/user/.local/bin"

# We love UTF!
ENV LANG C.UTF-8

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# Set the nvidia container runtime environment variables
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV CUDA_HOME="/usr/local/cuda"
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX 8.9"

# Install some handy tools. Even Guvcview for webcam support!
RUN set -x \
	&& apt-get update \
	&& apt-get install -y apt-transport-https ca-certificates \
	&& apt-get install -y git vim tmux nano htop sudo curl wget gnupg2 \
	&& apt-get install -y bash-completion \
	&& apt-get install -y guvcview \
	&& rm -rf /var/lib/apt/lists/* \
	&& useradd -ms /bin/bash user \
	&& echo "user:user" | chpasswd && adduser user sudo \
	&& echo "user ALL=(ALL) NOPASSWD: ALL " >> /etc/sudoers

RUN set -x \
    && apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN set -x \
    && apt-get update \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.11 python3.11-venv python3.11-dev \
    && apt-get install -y python3.11-tk


RUN sudo ln -s /usr/bin/python3.11 /usr/bin/python

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

WORKDIR /home/user

# Install SAM2
RUN git clone https://github.com/facebookresearch/segment-anything-2 && \
    cd segment-anything-2 && \
    python3.11 -m pip install -e . -v && \
    python3.11 -m pip install -e ".[demo]" && \
    cd checkpoints && ./download_ckpts.sh && cd ..


# dependency: lang-segment-anything
RUN apt update

# installing system dependencies:
RUN apt install -y git
RUN apt install libgl1-mesa-glx -y
RUN apt install libglib2.0-0 -y

# copy source code:
RUN git clone https://github.com/luca-medeiros/lang-segment-anything.git && \
    cd lang-segment-anything && \
    python3.11 -m pip install -e . -v && cd ..

RUN python3.11 -m pip install transformers==4.45.2 datasets==2.20.0 accelerate==0.26.1 jupyter==1.0.0 numpy pillow==10.2.0 peft==0.13.2 diffusers==0.30.3 timm==0.9.16

RUN python3.11 -m pip install hpsv2
COPY docker/bpe_simple_vocab_16e6.txt.gz /usr/local/lib/python3.11/dist-packages/hpsv2/src/open_clip/

RUN apt-get update
RUN python3.11 -m pip install pyfacer onnx2torch kornia omegaconf fvcore google-genai==1.20.0

RUN mkdir /home/user/multihuman-testbench
WORKDIR /home/user/multihuman-testbench

COPY . /home/user/multihuman-testbench/

RUN mkdir /home/user/multihuman-testbench/third_party
RUN git clone https://github.com/VectorSpaceLab/OmniGen.git /home/user/multihuman-testbench/third_party/OmniGen

ENV PYTHONPATH=/home/user/multihuman-testbench/:/home/user/multihuman-testbench/third_party/OmniGen
RUN apt-get update && apt-get install -y unzip

RUN cd data && unzip poses.zip
RUN chmod -R 755 /home/user


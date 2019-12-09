# FROM mcr.microsoft.com/azureml/o16n-base/python-assets@sha256:20ba3085141845301907d7ce6e9ee8388a0e43074f56c262d6de7efebf2ba98f AS inferencing-assets

# Tag: cuda:10.0-cudnn7-devel-ubuntu18.04
# Env: CUDA_VERSION=10.0.130
# Env: CUDA_PKG_VERSION=10-0=10.0.130-1
# Env: NCCL_VERSION=2.4.8
# Env: CUDNN_VERSION=7.6.3.30
# Env: NVIDIA_VISIBLE_DEVICES=all
# Env: NVIDIA_DRIVER_CAPABILITIES=compute,utility
# Env: NVIDIA_REQUIRE_CUDA=cuda>=10.0 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=410,driver<411
# Label: com.nvidia.cuda.version=10.0.130
# Label: com.nvidia.cudnn.version=7.6.3.30
# Label: com.nvidia.volumes.needed=nvidia_driver
# Ubuntu 18.04
FROM nvcr.io/nvidia/pytorch:19.11-py3

USER root:root

ENV com.nvidia.cuda.version $CUDA_VERSION
ENV com.nvidia.volumes.needed nvidia_driver
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND noninteractive
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
ENV NCCL_DEBUG=INFO
ENV HOROVOD_GPU_ALLREDUCE=NCCL

# Inference
# COPY --from=inferencing-assets /artifacts /var/

# Install Common Dependencies
RUN apt-get update && \
    # apt-get install -y --no-install-recommends \
    # --allow-change-held-packages \
    # # SSH and RDMA
    # libmlx4-1 \
    # libmlx5-1 \
    # librdmacm1 \
    # libibverbs1 \
    # libmthca1 \
    # libdapl2 \
    # dapl2-utils \
    # openssh-client \
    # openssh-server \
    # iproute2 && \
    # Others
    apt-get install -y --no-install-recommends \
    --allow-change-held-packages \
    build-essential \
    bzip2=1.0.6-8.1ubuntu0.2 \
    libbz2-1.0=1.0.6-8.1ubuntu0.2 \
    systemd \
    git=1:2.17.1-1ubuntu0.4 \
    wget \
    vim \
    tmux \
    unzip \
    htop \
    ca-certificates \
    libjpeg-dev \
    cpio \
    libsm6 \
    libxext6 \
    libxrender-dev \
    fuse && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Install lib for video
# RUN apt-get update && apt-get install -y software-properties-common
# RUN add-apt-repository -y ppa:jonathonf/ffmpeg-3
# RUN apt update && apt-get install -y libavformat-dev libavcodec-dev libswscale-dev libavutil-dev libswresample-dev
# RUN apt-get install -y ffmpeg
RUN export LIBRARY_PATH=/usr/local/lib:$LIBRARY_PATH

# Set timezone
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime


RUN conda install -y pyyaml scipy scikit-learn matplotlib pandas setuptools Cython h5py graphviz libgcc mkl-include cmake cffi typing
RUN conda clean -ya
RUN pip install boto3 addict tqdm regex pyyaml opencv-python azureml-defaults opencv-contrib-python nltk spacy future tensorboard sentencepiece
# Set CUDA_ROOT
RUN export CUDA_HOME="/usr/local/cuda"
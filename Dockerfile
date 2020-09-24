FROM mcr.microsoft.com/azureml/o16n-base/python-assets@sha256:20a8b655a3e5b9b0db8ddf70d03d048a7cf49e569c4f0382198b1ee77631a6ad AS inferencing-assets

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
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

USER root:root

ENV com.nvidia.cuda.version $CUDA_VERSION
ENV com.nvidia.volumes.needed nvidia_driver
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND noninteractive
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
ENV NCCL_DEBUG=INFO
ENV HOROVOD_GPU_ALLREDUCE=NCCL

# Inference
COPY --from=inferencing-assets /artifacts /var/

# Install Common Dependencies
RUN printf "deb http://archive.debian.org/debian/ jessie main\ndeb-src http://archive.debian.org/debian/ jessie main\ndeb http://security.debian.org jessie/updates main\ndeb-src http://security.debian.org jessie/updates main" > /etc/apt/sources.list
RUN apt-get update && \
    apt-get install -y apt-transport-https && \
    apt-get install -y --no-install-recommends \
    # SSH and RDMA
    libmlx4-1 \
    libmlx5-1 \
    librdmacm1 \
    libibverbs1 \
    libmthca1 \
    libdapl2 \
    dapl2-utils \
    openssh-client \
    openssh-server \
    iproute2 && \
    # Others
    apt-get install -y --no-install-recommends \
    --allow-change-held-packages \
    build-essential \
    bzip2=1.0.6-8.1ubuntu0.2 \
    libbz2-1.0=1.0.6-8.1ubuntu0.2 \
    systemd \
    git \
    libnccl2=2.5.6-1+cuda10.1 \
    libnccl-dev=2.5.6-1+cuda10.1 \
    wget \
    vim \
    tmux \
    unzip \
    ca-certificates \
    libjpeg-dev \
    cpio \
    jq \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    fuse && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Install lib for video
# RUN apt-get update && apt-get install -y software-properties-common
# RUN add-apt-repository -y ppa:jonathonf/ffmpeg-3
# RUN apt update && apt-get install -y libavformat-dev libavcodec-dev libswscale-dev libavutil-dev libswresample-dev
# RUN apt-get install -y ffmpeg
# RUN export LIBRARY_PATH=/usr/local/lib:$LIBRARY_PATH

# Set timezone
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

# Inference
# Copy logging utilities, nginx and rsyslog configuration files, IOT server binary, etc.
COPY --from=inferencing-assets /artifacts /var/
RUN /var/requirements/install_system_requirements.sh && \
    cp /var/configuration/rsyslog.conf /etc/rsyslog.conf && \
    cp /var/configuration/nginx.conf /etc/nginx/sites-available/app && \
    ln -s /etc/nginx/sites-available/app /etc/nginx/sites-enabled/app && \
    rm -f /etc/nginx/sites-enabled/default
ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=300
EXPOSE 5001 8883 8888

# Conda Environment
ENV MINICONDA_VERSION latest
ENV PATH /opt/miniconda/bin:$PATH
RUN wget -qO /tmp/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -bf -p /opt/miniconda && \
    conda clean -ay && \
    rm -rf /opt/miniconda/pkgs && \
    rm /tmp/miniconda.sh && \
    find / -type d -name __pycache__ | xargs rm -rf
# Open-MPI installation
ENV OPENMPI_VERSION 3.1.2
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://download.open-mpi.org/release/open-mpi/v3.1/openmpi-${OPENMPI_VERSION}.tar.gz && \
    tar zxf openmpi-${OPENMPI_VERSION}.tar.gz && \
    cd openmpi-${OPENMPI_VERSION} && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi


RUN conda install -y python=3.6 numpy pyyaml scipy ipython mkl scikit-learn matplotlib pandas setuptools Cython h5py graphviz libgcc mkl-include cmake cffi typing cython && \
     conda install -y -c mingfeima mkldnn && \
     conda install -c anaconda gxx_linux-64
RUN conda clean -ya
RUN pip install boto3 addict tqdm regex pyyaml opencv-python opencv-contrib-python nltk spacy future tensorboard wandb filelock tokenizers sentencepiece ruamel.yaml
# Set CUDA_ROOT
RUN export CUDA_HOME="/usr/local/cuda"

# Install pytorch
RUN conda install pytorch==1.4.0 torchvision cudatoolkit=10.1 -c pytorch
#Install Faiss
RUN conda install faiss-gpu -c pytorch # For CUDA10.1

# Install horovod
# RUN HOROVOD_GPU_ALLREDUCE=NCCL pip install --no-cache-dir horovod==0.16.1

#Install apex
RUN pip uninstall -y apex || :
RUN cd /tmp && \
    SHA=ToUcHMe git clone https://github.com/NVIDIA/apex.git
RUN cd /tmp/apex/ && \
    python setup.py install --cuda_ext --cpp_ext && \
    rm -rf /tmp/apex*

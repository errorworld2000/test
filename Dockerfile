# 使用基础镜像
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# 更新系统并安装必要的依赖项
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    software-properties-common \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-venv \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y python3.12 python3.12-venv python3.12-dev \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# 安装 PyTorch 和 torchvision
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装 JupyterLab
RUN pip install jupyterlab

# 安装 poetry
RUN pip install poetry

# 配置工作目录
WORKDIR /workspace

# 克隆代码仓库
RUN git clone https://github.com/errorworld2000/test.git /workspace

# 切换到项目目录
WORKDIR /workspace

# 安装项目依赖
RUN poetry install --no-root

# 暴露 JupyterLab 端口
RUN pip install jupyterlab
EXPOSE 8888

# 启动 JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
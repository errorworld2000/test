# 使用基础镜像
FROM jungleer/pytorch_env:latest

# 配置工作目录
WORKDIR /workspace

# 克隆代码仓库
RUN git clone https://github.com/errorworld2000/test.git /workspace

# 切换到项目目录
WORKDIR /workspace

# 安装项目依赖
RUN poetry config virtualenvs.create false && poetry install --no-root

# 暴露 JupyterLab 端口
EXPOSE 8888

# 启动 JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
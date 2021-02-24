FROM snowind/cuda-python:0.2.0-prelude
LABEL maintainer = "Nemo <nemo.tao@refinedchina.com>" \
      description = "Attention Monitoring System Analysis Engine Service"

COPY requirements.txt /
COPY entrypoint.sh /

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip install -r /requirements.txt && \
    mkdir /project && \
    chmod +x /entrypoint.sh

COPY src/ /project

WORKDIR /project

ENTRYPOINT '/entrypoint.sh'
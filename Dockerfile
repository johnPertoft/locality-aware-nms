#FROM tensorflow/tensorflow:2.2.0-custom-op-ubuntu16

#RUN git clone https://github.com/tensorflow/custom-op.git /custom-op
#WORKDIR /locality-aware-nms
#RUN cp -r /custom-op/tf /locality-aware-nms/tf

FROM ubuntu:18.04

RUN apt-get update && apt-get install -y \
    curl \
    git \
    gnupg \
    rsync \
    software-properties-common

RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y \
    python3.6 \
    python3.7 \
    python3.8
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.6 get-pip.py && \
    python3.7 get-pip.py && \
    python3.8 get-pip.py

RUN \
    curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg && \
    mv bazel.gpg /etc/apt/trusted.gpg.d/ && \
    echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list
RUN apt-get update && apt-get install -y \
    bazel=2.0.0

RUN git clone https://github.com/tensorflow/custom-op.git /custom-op
WORKDIR /locality-aware-nms
RUN cp -r /custom-op/tf /locality-aware-nms/tf

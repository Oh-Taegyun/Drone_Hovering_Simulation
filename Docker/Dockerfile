FROM python:3.12-slim

# 필수 시스템 라이브러리 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        build-essential \
        git \
        libx11-6 \
        libgl1 \
        libxrandr2 \
        libxinerama1 \
        libxcursor1 \
        libxi6 \
        libglib2.0-0 \    
        && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio && \
    pip install genesis-world tqdm && \
    pip install open3d

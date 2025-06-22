FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip git ffmpeg build-essential unrar \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip
RUN pip install faster-whisper ctranslate2[cuda] rarfile

WORKDIR /app
COPY . .

CMD ["python3", "processarVideos.py"]
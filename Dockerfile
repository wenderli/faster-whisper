# Base com suporte CUDA + Python
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Instala dependências do sistema
RUN apt-get update && apt-get install -y \
    python3 python3-pip git ffmpeg build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Atualiza pip e instala dependências Python
RUN python3 -m pip install --upgrade pip

# Instala o pacote com suporte à GPU
RUN pip install faster-whisper ctranslate2[cuda] 

# Cria diretório de trabalho
WORKDIR /app

# Copia os arquivos do projeto
COPY . .

# Comando padrão (pode ser ajustado)
CMD ["python3", "transcribe.py"]

#FROM deepspeed/deepspeed:v072_torch112_cu117
FROM ghcr.io/coqui-ai/tts
WORKDIR /app

# Install linux audio package
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y libsndfile1

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-cuda-toolkit

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y libaio-dev

# Update pip
RUN pip install --upgrade pip

# Install Python requirements
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy over app contents
COPY . .

ENV COQUI_TOS_AGREED = 1
# Preload model
# COPY models/ /root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v1.1

# Set environemnt variables
ENV READER_PORT $READER_PORT
ENV BM_TO_TTS_REDIS_HOST "192.168.1.75"

# Start a terminal
#CMD ["--list_models"]

# Start uvicorn api
# uvicorn main:app --host 0.0.0.0 --port 8100

#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8100"]
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8100"]
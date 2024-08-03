#FROM deepspeed/deepspeed:v072_torch112_cu117
FROM ghcr.io/coqui-ai/tts
WORKDIR /app

EXPOSE 22

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
#COPY models/tts_models--multilingual--multi-dataset--xtts_v1.1 /root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v1.1
COPY model_cache/tts_models--multilingual--multi-dataset--xtts_v2.0.2  /root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2.0.2

# Install OpenSSH server
RUN apt-get update && \
    apt-get install -y openssh-server

RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1000 test

RUN echo 'test:test' | chpasswd

RUN service ssh start


# Set environemnt variables
ENV READER_PORT $READER_PORT

# Start a terminal
#CMD ["--list_models"]

# Start uvicorn api
# uvicorn main:app --host 0.0.0.0 --port 8100

#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8100"]
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8100"]
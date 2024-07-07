FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04


ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Set work directory
WORKDIR /app
RUN apt-get update && apt-get install -y \
wget \
ffmpeg \
python3.8 \
python3.8-venv \
python3.8-dev \
git \
&& apt-get clean

# Install pip for Python 3.9
RUN wget https://bootstrap.pypa.io/get-pip.py && python3.8 get-pip.py

COPY pip.conf /root/.pip/pip.conf


# Set Python 3.9 and pip3.9 as the default python3 and pip3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
RUN update-alternatives --install /usr/bin/pip3 pip3 /usr/local/bin/pip3.8 1

# Upgrade pip and install pip packages
#RUN python3.9 -m ensurepip
RUN pip3 install --no-cache-dir --upgrade pip
#RUN pip install ffmpeg-python torch flask
RUN pip install torch==1.12.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torchaudio==0.12.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install ffmpeg
#run pip install numpy==1.23.4 opencv-python tqdm Pillow python-magic
run pip install opencv-python Pillow python-magic
#run pip install safetensors scipy scikit-image pandas 
#run pip install kornia facexlib 
#run pip install yacs pydub gfpgan
#run pip install librosa

RUN git clone https://github.com/cpippenger/SadTalker-micro.git
# THIS VERSION ONLY!

#RUN pip install -r SadTalker-micro/requirements.txt 

#RUN SERVICE_DEBUG=False \
# DATABASE_SCHEMA="chat_cp" \
# SERVICE_PORT=6667 \
# SERVICE_HOST="localhost" \
# python3.9 SadTalker-micro/flask_inference.py

ENV SERVICE_DEBUG=False 
ENV DATABASE_SCHEMA="chat_cp" 
ENV SERVICE_PORT=7666 
ENV SERVICE_HOST="localhost" 
#python3.9 SadTalker-micro/flask_inference.py

#export SERVICE_DEBUG=False \
#export DATABASE_SCHEMA="chat_cp" \
#export SERVICE_PORT=6667 \
#export SERVICE_HOST="localhost" \
#python3.9 SadTalker-micro/flask_inference.py
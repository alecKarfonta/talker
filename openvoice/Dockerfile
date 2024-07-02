#FROM python:3.9-slim
#FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04
#FROM nvcr.io/nvidia/pytorch:21.10-py3
FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04


ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Set work directory
WORKDIR /app

# Install git and other dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    nano \
    software-properties-common \
    && apt-get clean

# Add deadsnakes PPA and install Python 3.9
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get install -y \
python3.9 \
python3.9-venv \
python3.9-dev \
&& apt-get clean

# Install pip for Python 3.9
RUN wget https://bootstrap.pypa.io/get-pip.py && python3.9 get-pip.py

# Set Python 3.9 and pip3.9 as the default python3 and pip3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
RUN update-alternatives --install /usr/bin/pip3 pip3 /usr/local/bin/pip3.9 1

# Upgrade pip and install pip packages
RUN python3.9 -m ensurepip
RUN pip3 install --no-cache-dir --upgrade pip

# Install git
RUN apt-get update && apt-get install -y git && apt-get clean

# Install dependencies
COPY pip.conf /root/.pip/pip.conf

COPY requirements.txt requirements.txt

RUN pip3.9 install --no-cache-dir -r requirements.txt

RUN pip3.9 install torch==2.2.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
#torchvision==0.11.2+cu112 torchaudio==0.10.1+cu112 

# Copy the application files
COPY . .

#RUN git clone git@github.com:myshell-ai/OpenVoice.git

RUN pip3.9 install -e OpenVoice
RUN pip3.9 install git+https://github.com/myshell-ai/MeloTTS.git



#RUN pip3 install -e MeloTTS

#RUN pip install git+https://github.com/myshell-ai/MeloTTS.git

RUN python3.9 -m unidic download


# Expose the port the app runs on
EXPOSE 5001

# Run the application
#CMD ["python3", "app.py"]
CMD ["./startup.sh"]
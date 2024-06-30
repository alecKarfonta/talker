#FROM python:3.9-slim
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

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

# Set Python 3.9 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
RUN update-alternatives --install /usr/bin/pip3 pip3 /usr/bin/pip3 1

# Upgrade pip and install pip packages
RUN python3.9 -m ensurepip
RUN pip3 install --no-cache-dir --upgrade pip

# Install git
RUN apt-get update && apt-get install -y git && apt-get clean

# Install dependencies
COPY pip.conf /root/.pip/pip.conf

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY . .

#RUN git clone git@github.com:myshell-ai/OpenVoice.git

RUN pip install -e OpenVoice
RUN pip install -e MeloTTS

#RUN pip install git+https://github.com/myshell-ai/MeloTTS.git

RUN python -m unidic download


# Expose the port the app runs on
EXPOSE 5001

# Run the application
CMD ["python", "app.py"]
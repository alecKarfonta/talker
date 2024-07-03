# Use the official Python image from the Docker Hub
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install required dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    build-essential \
    cmake \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file
COPY requirements.txt .

COPY pip.conf /root/.pip/pip.conf

# Copy the rest of the application code
COPY . .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables for CUDA
ENV CUDAToolkit_ROOT /usr/local/cuda

RUN CMAKE_ARGS="-DGGML_CUDA=ON" FORCE_CMAKE=1 pip install llama-cpp-python

#RUN wget https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q4_0.gguf?ref=localhost

#RUN mv zephyr-7b-beta.Q4_0.gguf?ref=localhost zephyr-7b-beta.Q4_0.gguf

# Install OpenSSH server
RUN apt-get update && \
    apt-get install -y openssh-server

RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1000 test

RUN echo 'test:test' | chpasswd

RUN service ssh start


# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]


# Copy the startup script
COPY start.sh /app/start.sh

# Make the startup script executable
RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
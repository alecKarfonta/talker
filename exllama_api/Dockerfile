# Use the official Python image from the Docker Hub
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install required dependencies and Python 3.11
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    wget \
    mlocate \
    nano \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.10 python3.10-dev python3.10-venv \
    && wget https://bootstrap.pypa.io/get-pip.py \
    && python3.10 get-pip.py \
    && rm get-pip.py \
    && rm -rf /var/lib/apt/lists/*


# Set Python 3.11 as the default python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --set python3 /usr/bin/python3.10
# Set the working directory
WORKDIR /app

# Copy the requirements.txt file
COPY requirements.txt .

COPY pip.conf /root/.pip/pip.conf

# Copy the rest of the application code
COPY . .

#RUN git clone https://github.com/turboderp/exllamav2

# Install the dependencies
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Clone and install ExLlamaV2
#RUN git clone https://github.com/turboderp/exllamav2
#RUN python3 -m pip install -r exllamav2/requirements.txt
#RUN python3 -m pip install exllamav2/.
#RUN pip3 intall exllamav2


# Install the dependencies
#RUN pip install --no-cache-dir -r requirements.txt
# Set environment variables for CUDA
#ENV CUDAToolkit_ROOT /usr/local/cuda
#RUN CMAKE_ARGS="-DGGML_CUDA=ON" FORCE_CMAKE=1 pip install llama-cpp-python

#RUN wget https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q4_0.gguf?ref=localhost

#RUN mv zephyr-7b-beta.Q4_0.gguf?ref=localhost zephyr-7b-beta.Q4_0.gguf

# Install OpenSSH server
RUN apt-get update && \
    apt-get install -y openssh-server

RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1000 test

RUN echo 'test:test' | chpasswd

RUN service ssh start

EXPOSE 22


# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]


# Copy the startup script
COPY start.sh /app/start.sh

# Make the startup script executable
RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
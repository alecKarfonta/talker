FROM python:3.9

WORKDIR /app

EXPOSE 22

COPY requirements.txt .

# Copy pip config
COPY pip.conf /root/.pip/pip.conf

RUN pip install -r requirements.txt

COPY . .

RUN chmod 777 /app/startup.sh

# Install transformers from source
#RUN pip install "git+https://github.com/huggingface/transformers.git"

# Install accelerate from source
#RUN pip install git+https://github.com/huggingface/accelerate

#RUN pip install -i https://pypi.org/simple/ bitsandbytes
#RUN pip install accelerate
#RUN pip install optimum


# Install auto-gptq from source
#RUN pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/
# Install bitsnbytes
#RUN pip install bitsandbytes
# Set cuda path for bitsandbytes
#RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib/libcudart.so

#RUN pip install datasets
#RUN pip install pydub
#RUN pip install soundfile



# Install OpenSSH server
RUN apt-get update && \
    apt-get install -y openssh-server

RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1000 test

RUN echo 'test:test' | chpasswd

RUN service ssh start


#CMD ["python", "app.py"]
CMD ["./startup.sh"]

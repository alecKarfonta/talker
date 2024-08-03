FROM python:3.9

WORKDIR /app

EXPOSE 22

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

RUN chmod 777 /app/startup.sh

# Install OpenSSH server
RUN apt-get update && \
    apt-get install -y openssh-server

RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1000 test

RUN echo 'test:test' | chpasswd

RUN service ssh start


#CMD ["python", "app.py"]
CMD ["./startup.sh"]

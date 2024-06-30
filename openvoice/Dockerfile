FROM python:3.9-slim

# Set work directory
WORKDIR /app

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
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
#!/bin/bash

# Start the SSH service
service ssh start

# Start the uvicorn server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
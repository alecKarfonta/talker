#!/bin/bash

# Stop and remove the existing container if it exists
if [ "$(docker ps -a -q -f name=llama-api)" ]; then
    echo "Stopping and removing existing container..."
    docker stop llama-api
    docker rm llama-api
fi

# Run the new container
echo "Running the new container..."
docker run -d --name llama-api -p 8400:8000 -p 2223:22 --gpus '"device=0"' llama-api
        

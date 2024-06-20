#!/bin/bash

# Stop and remove the existing container if it exists
if [ "$(docker ps -a -q -f name=text-generation-container)" ]; then
    echo "Stopping and removing existing container..."
    docker stop text-generation-container
    docker rm text-generation-container
fi

# Run the new container
echo "Running the new container..."
docker run -d --name text-generation-container -p 8400:8000 -p 2223:22 --gpus '"device=0"' text-generation-api
        

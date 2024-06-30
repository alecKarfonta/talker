#!/bin/bash

# Stop and remove the existing container if it exists
if [ "$(docker ps -a -q -f name=tts-api)" ]; then
    echo "Stopping and removing existing container..."
    docker stop tts-api
    docker rm tts-api
fi

# Run the new container
echo "Running the new container..."
docker run -it --name tts-api -p 8001:5001 -p 2225:22 --gpus '"device=0"' --entrypoint '/bin/bash' tts-api
        

#!/bin/bash

# Stop and remove the existing container if it exists
if [ "$(docker ps -a -q -f name=tts_api)" ]; then
    echo "Stopping and removing existing container..."
    docker stop tts_api
    docker rm tts_api
fi

# Run the new container
echo "Running the new container..."
docker run -it --name tts_api -p 8001:5001 -p 2223:22 --gpus '"device=0"' --entrypoint '/bin/bash' tts_api:1.0
        

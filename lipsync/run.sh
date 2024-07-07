#!/bin/bash

# Stop and remove the existing container if it exists
if [ "$(docker ps -a -q -f name=lipsync)" ]; then
    echo "Stopping and removing existing container..."
    docker stop lipsync
    docker rm lipsync
fi

# Run the new container
echo "Running the new container..."
docker run -it --name lipsync -p 7666:7666 --gpus '"device=0"' --entrypoint /bin/bash lipsync 
        

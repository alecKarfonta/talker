#!/bin/bash

# Stop and remove the existing container if it exists
if [ "$(docker ps -a -q -f name=xtalker)" ]; then
    echo "Stopping and removing existing container..."
    docker stop xtalker
    docker rm xtalker
fi

# Run the new container
echo "Running the new container..."
docker run -it --name xtalker -p 7666:7666 --gpus '"device=1"' --entrypoint /bin/bash xtalker 
        

docker run --rm \
        --name reader \
        --gpus all \
        --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        --runtime=nvidia -it \
        --publish 8100:8100 \
        -e READER_PORT='8100' \
        -e COQUI_TOS_AGREED=1 \
        -w /app \
        reader

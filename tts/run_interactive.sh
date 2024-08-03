docker run --rm \
        --name reader \
        --gpus all \
        --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        --publish 8100:8100 \
        --publish 2223:22 \
        -e READER_PORT='8100' \
        -e COQUI_TOS_AGREED=1 \
        -w /app \
        -it \
        --entrypoint /bin/bash \
        reader

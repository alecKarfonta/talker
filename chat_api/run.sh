sudo docker run \
   --gpus all \
   --ipc=host \
   --ulimit memlock=-1 \
   --ulimit stack=67108864 \
   --publish 5001:5001 \
   --publish 2224:22 \
   -e DATABASE_HOST='192.168.1.4' \
   -e SERVICE_PORT='5001' \
   -it \
   chat_api:1.0 /bin/bash
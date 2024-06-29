sudo docker run \
   --ipc=host \
   --ulimit memlock=-1 \
   --ulimit stack=67108864 \
   --publish 5001:5001 \
   --publish 2224:22 \
   -e DATABASE_HOST='threadripper' \
   -e SERVICE_PORT='5001' \
   -it \
   chat_api:1.0 /bin/bash
sudo docker run \
   --gpus all \
   --ipc=host \
   --ulimit memlock=-1 \
   --ulimit stack=67108864 \
   --publish 8001:5001 \
   --publish 2225:22 \
   -e DATABASE_HOST='threadripper' \
   -it \
   tts_api:1.0 /bin/bash
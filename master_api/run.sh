sudo docker run \
   --ipc=host \
   --ulimit memlock=-1 \
   --ulimit stack=67108864 \
   --publish 5001:5001 \
   --publish 2221:22 \
   -e DATABASE_HOST='192.168.1.4' \
   -e SERVICE_PORT='5000' \
   -it \
   master_api:1.0 /bin/bash
# Build image
sudo docker build --tag py:1.0 . -f /home/code/Dockerfile_pytorch

# Start image
sudo nvidia-docker run -ti --rm --publish 8888:8888 -p 5900:5900 -v /home/alec/git:/py/git -w /py py:1.0
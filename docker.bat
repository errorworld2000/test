docker build -f Dockerfile -t test:latest . 
docker run --name test1 --gpus all -it -v ${PWD}:/workspace -p 8888:8888 test
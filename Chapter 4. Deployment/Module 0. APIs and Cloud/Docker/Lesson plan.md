# Docker

## What is Docker?
Docker allows you to make **containers** from **images**.
These images run  virtual operating systems on top of whatever OS is actually installed on the machine running the container.

## The Dockerfile
A Dockerfile contains a sequence of commands that a user could run to creat a docker image
- get image
- copy app folder from prev module
- run the app.py

## Making a Docker image
- run ```docker build -t <tag> . ``` to build the image and tag it

## Creating an instance of a docker image - a docker container
- run ```docker run --name <container_name> <tag>```

## See Docker containers
- run docker ps
- run docker ps -a

## Stop the Docker container
- run ```docker stop <container_name> --time=0```

## Run the container interactively and with a terminal

## Port mapping
- what the hell is a port?

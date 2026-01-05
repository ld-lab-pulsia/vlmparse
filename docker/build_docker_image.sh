#!/bin/bash
UNAME=$(whoami)
. constants.env

docker build -t $IMAGE_NAME:latest . --file docker/Dockerfile --progress plain

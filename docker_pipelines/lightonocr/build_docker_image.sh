#!/bin/bash
UNAME=$(whoami)

docker build -t docparser:latest . --file docker/Dockerfile --progress plain
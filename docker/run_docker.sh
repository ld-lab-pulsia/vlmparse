#!/bin/bash
UNAME=$(whoami)

. constants.env

docker run -d -it --rm --runtime=nvidia \
        --name $IMAGE_NAME-$UNAME-dev \
        --cpus $CPUS \
        -v $PWD:/workspace \
        -v $DATA_DIR_DOCKER:$DATA_DIR \
        -p $SSH_PORT:$SSH_PORT \
        -e HF_HOME=/data/models/hub/ \
        -w /workspace \
        --shm-size="256g"\
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v /usr/bin/docker:/usr/bin/docker \
        $IMAGE_NAME:latest bash

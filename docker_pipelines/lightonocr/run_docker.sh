#!/bin/bash
UNAME=$(whoami)

. constants.env

docker run -d --rm --runtime=nvidia \
        --name $IMAGE_NAME-$UNAME-dev \
        --cpus $CPUS \
        -v $PWD:/workspace \
        -v $DATADIR_PROJECT:/data_project \
        -v $DATADIR_DEVICE:/data_device \
        -v $MODELDIR:/models \
        -v /workspace/docparser/.venv \
        -p $SSH_PORT:22 \
        -p $STREAMLIT_PORT:8501 \
        -p $API_PORT:8000 \
        -w /workspace \
        --shm-size="256g"\
        $IMAGE_NAME:latest

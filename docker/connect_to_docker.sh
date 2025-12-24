#!/bin/bash
. constants.env

docker exec -it --user $(id -u $USER):$(id -g) $IMAGE_NAME-$USER-dev bash

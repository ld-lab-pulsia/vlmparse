#!/bin/bash
. constants.env
docker exec -it --user root $IMAGE_NAME-$USER-dev bash

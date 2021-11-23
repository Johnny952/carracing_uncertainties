#!/bin/bash

docker run --name carracing --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 -it -v $(pwd):/home/user/workspace carracing /bin/bash
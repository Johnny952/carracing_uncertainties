#!/bin/bash

docker run --name carracing --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=5 -it -v $(pwd):/home/user/workspace carracing /bin/bash
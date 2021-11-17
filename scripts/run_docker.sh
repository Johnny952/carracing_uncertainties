#!/bin/bash

docker run --name carracing --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=5 -it -v ~/cachefs/sync:/home/user/workspace/sync carracing /bin/bash
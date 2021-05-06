#!/bin/bash

docker run --name carracing --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=5 -v -it carracing /bin/bash
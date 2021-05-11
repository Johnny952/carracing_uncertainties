#!/bin/bash

docker run --name carracing --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=5 -it carracing /bin/bash
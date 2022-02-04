#!/bin/bash

docker run --name carracing-server --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 \
    -it -v $(pwd):/home/user/workspace \
    -p 8000:8000 carracing-server uvicorn main:app --reload --host 0.0.0.0 --port 8000
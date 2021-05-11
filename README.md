# Car Racing with PyTorch
Solving ```CarRacing-v0``` problem from OpenAI using Proximal Policy Optimization and Deep Q-Learning. PPO implementation is based on https://github.com/xtma/pytorch_car_caring script.

This implementation is done over Docker and VS Code, and you don't need to install anything else on your local machine.

## Requirement
The required libraries used can be seen in ```requirements.txt``` and ```Dockerfile```, in the last file is installed mujoco, gym and every dependecy needed for them.

The main dependencies are:

- [pytorch == 1.8.1](https://pytorch.org/)
- [gym == 0.17.2](https://github.com/openai/gym)
- [wandb == 0.10.29](https://wandb.ai)
- [blitz-bayesian-pytorch == 0.2.7](https://github.com/piEsposito/blitz-bayesian-deep-learning)

## Image and Container creation
To build the image you can run directly ```sh scripts/build_docker.sh``` or execute the following command, where ```-t``` is the image tag name.

    docker build -t <image name> .


To create container run ```sh scripts/run_docker``` or execute the following command, where ```NVIDIA_VISIBLE_DEVICES=``` is the gpu number to use or gpus, change to 0 if you have only one:

    docker run --name <container name> --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=<gpu/s number> -it <image name> /bin/bash


## Training
You can start train the model getting inside the model folder you want to train and run train script, for example:

    cd ppo/
    python train.py

There are many parameters, run ```python train.py -h``` to see them all and change them.

To use wandb, you have to add a file ```config.json``` at same level as ```train.py``` in ppo or dqn with the following:

    {
        "project": <project name>,
        "entity": <entity name>
    }


## Performance

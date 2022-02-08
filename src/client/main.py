from websocket import create_connection
import json
import numpy as np
import argparse
from manual_drive import main

import sys
sys.path.append('..')
from shared.components.env import Env

# websocket.enableTrace(True)
class Model(object):
    def __init__(self, env, nb_episodes, path="ws://localhost:8000/ws", manual_drive=False, uncertainties=False, **kwargs) -> None:
        self._manual_drive = manual_drive
        self._uncertainties = uncertainties
        self._env = env
        self.nb_episodes = nb_episodes

        self._use_api = not manual_drive or uncertainties
        if self._use_api:
            raise NotImplementedError()
            self.ws = create_connection(path)
            self.ws.send(json.dumps(kwargs))
            result =  self.ws.recv()
            success = json.loads(result)["success"]
            if not success:
                raise Exception("Model could not be builded")
    
    def send_img(self, image):
        array_shape = image.shape
        array_data_type = image.dtype.name
        array_string = image.tostring()
        to_send = {
            "shape": array_shape,
            "dtype": array_data_type,
            "img": array_string.decode("latin-1"),
        }
        self.ws.send(json.dumps(to_send))
        result =  self.ws.recv()
        print(json.loads(result))
    
    def run(self):
        if not self._use_api:
            main(self._env, self.nb_episodes)
        else:
            self.close()

    def close(self):
        self.ws.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Server-Client Testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-NE",
        "--nb-episodes",
        type=int,
        default=1,
        help="Number of episodes to test",
    )
    parser.add_argument(
        "-IS",
        "--image-stack",
        type=int,
        default=4,
        help="Number of images to stack as observation",
    )
    parser.add_argument(
        "-AR",
        "--action-repeat",
        type=int,
        default=1,
        help="Times the action repeats",
    )
    parser.add_argument(
        "-MD",
        "--manual-drive",
        action="store_true",
        help="Whether manually drive the car or not",
    )
    parser.add_argument(
        "-U",
        "--uncertainties",
        action="store_true",
        help="Whether plot uncertainties in real time or not",
    )
    args = parser.parse_args()

    env = Env(args.image_stack, args.action_repeat, evaluation=True)

    model = Model(
        env,
        args.nb_episodes,
        manual_drive=args.manual_drive,
        uncertainties=args.uncertainties,
    )
    model.run()
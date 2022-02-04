from websocket import create_connection
import json
import numpy as np
import time

# websocket.enableTrace(True)
class Model(object):
    def __init__(self, path="ws://localhost:8000/ws", **kwargs) -> None:
        self.ws = create_connection(path)
        #self.ws.send(json.dumps(kwargs))
        #result =  self.ws.recv()
        #print(json.loads(result))
    
    def send_img(self):
        img = np.random.rand(96, 96, 4)
        ding = time.time()
        array_shape = img.shape
        array_data_type = img.dtype.name
        array_string = img.tostring()
        to_send = {
            "shape": array_shape,
            "dtype": array_data_type,
            "img": array_string.decode("latin-1"),
        }
        self.ws.send(json.dumps(to_send))
        result =  self.ws.recv()
        print(json.loads(result))
        print(f"Dt: {1/(time.time() - ding)}hz")

    def close(self):
        self.ws.close()

if __name__ == "__main__":
    model = Model(lr=0.001, eval_every=10)
    model.send_img()
    model.close()
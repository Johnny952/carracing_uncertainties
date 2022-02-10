from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import json
import numpy as np
from utils.string2image import string2image

import sys
sys.path.append('..')
from ppo.components.agent import Agent

app = FastAPI()

# TODO: def build_ppo_agent

# TODO: def predict

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # agent = build_agent()
            await websocket.send_json({"success": True})




            o = string2image(json.loads(data))
            #await websocket.send_json({"message": json.loads(data)})
            await websocket.send_json({"message": "Done"})
    except WebSocketDisconnect:
        print("Websocket Connection closed")
    else:
        await websocket.close()

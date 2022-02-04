from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import json
import numpy as np

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            d = json.loads(data)
            dtype = d['dtype']
            shape = d['shape']
            img = np.fromstring(d['img'].encode('latin-1'), dtype=dtype).reshape(shape)
            #await websocket.send_json({"message": json.loads(data)})
            await websocket.send_json({"message": "Done"})
    except WebSocketDisconnect:
        print("Websocket Connection closed")
    else:
        await websocket.close()

from dataclasses import dataclass
import io
from typing import NamedTuple

import numpy
import torch
from ai.vision.VisionSystem import VisionSystem
from ai.vision.VideoLoader import VideoFolderSource
import time
from ai.vision import video_config
from fastapi import FastAPI, Response
from threading import Thread
from ai.vision.image_util import get_filtered_scaled
import chassis
import uvicorn
from PIL import Image

class ActionTensor(NamedTuple):
    drive: float
    steer: float


@dataclass
class SharedState:
    action_tensor: ActionTensor
    latent_space: torch.FloatTensor
    camera_image: numpy.ndarray


shared_state = SharedState(
    action_tensor=ActionTensor(0, 0),
    latent_space=torch.zeros(121).float(),
    camera_image=numpy.zeros((640, 480, 3), dtype=numpy.uint8)
)


vision_system = VisionSystem()

def ai_thread():
    vid = VideoFolderSource(video_config.VIDEO_SOURCES)

    vid.step()

    t = 0
    prev_time = time.time()
    while True:
        t += 1
        vid.step()
        new_frame = vid.get_full()
        latent_space = vision_system.ingest_frame(new_frame)

        # chassis.drive(shared_state.action_tensor.drive, shared_state.action_tensor.steer)
        shared_state.camera_image = get_filtered_scaled(new_frame, video_config.AI_RESOLUTION)
        shared_state.latent_space = latent_space
 
        # Print FPS
        if t % 10 == 0:
            new_time = time.time()
            print(f"FPS: {10 / (new_time - prev_time)}")
            prev_time = new_time




app = FastAPI()




def numpy_array_to_response_image(array: numpy.ndarray) -> Response:
    im = Image.fromarray(array)

    # save image to an in-memory bytes buffer
    with io.BytesIO() as buf:
        im.save(buf, format='PNG')
        im_bytes = buf.getvalue()

    headers = {'Content-Disposition': 'inline; filename="test.png"'}
    return Response(im_bytes, headers=headers, media_type='image/png')




@app.on_event("startup")
async def startup_event():
    thread = Thread(target=ai_thread)
    thread.start()



@app.get("/camera/full")
def get_camera_image():
    return numpy_array_to_response_image(shared_state.camera_image)

@app.get("/camera/reconstructed")
def get_reconstructed_image():
    reconstructed = vision_system.vision_model.decode_frames([shared_state.latent_space])[0]
    return numpy_array_to_response_image(reconstructed)

INDEX = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Car</title>
</head>
<body>
    <h1>AI Car</h1>
    <img id="full" src="/camera/full" />
    <img id="reconstructed" src="/camera/reconstructed" />
</body>
</html>

<script>
    setInterval(() => {
        document.querySelector("#reconstructed").src = "/camera/reconstructed?" + Date.now();
    }, 100);
    setInterval(() => {
        document.querySelector("#full").src = "/camera/full?" + Date.now();
    }, 100);
</script>
"""

@app.get("/")
def get_index():
    return Response(INDEX, media_type="text/html")


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)



if __name__ == "__main__":
    ai_thread()




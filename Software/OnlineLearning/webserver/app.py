from dataclasses import dataclass
import threading
from typing import NamedTuple, cast
from fastapi import Depends, FastAPI, Response
import numpy
import io
from PIL import Image
from pydantic import BaseModel
import torch

from ai.vision.VisionSystem import VisionSystem
from ai.vision.image_util import NumpyImage


class ActionTensor(NamedTuple):
    drive: float
    steer: float


@dataclass
class SharedState:
    action_tensor: ActionTensor
    latent_space: torch.Tensor
    camera_image: NumpyImage
    vision_system: VisionSystem | None


SHARED_STATE_MUTEX = threading.Lock()
SHARED_STATE = SharedState(
    action_tensor=ActionTensor(0, 0),
    latent_space=torch.zeros(121),
    camera_image=numpy.zeros((640, 480, 3), dtype=numpy.uint8),
    vision_system=None,
)


def numpy_array_to_response_image(array: numpy.ndarray) -> Response:
    im = Image.fromarray(array)

    # save image to an in-memory bytes buffer
    with io.BytesIO() as buf:
        im.save(buf, format="PNG")
        im_bytes = buf.getvalue()

    headers = {"Content-Disposition": 'inline; filename="test.png"'}
    return Response(im_bytes, headers=headers, media_type="image/png")


def shared_state():
    with SHARED_STATE_MUTEX:
        yield SHARED_STATE


app = FastAPI()


@app.get("/camera/full")
def get_camera_image(shared_state: SharedState = Depends(shared_state)):
    return numpy_array_to_response_image(shared_state.camera_image)


@app.get("/camera/reconstructed")
def get_reconstructed_image(shared_state: SharedState = Depends(shared_state)):
    if shared_state.vision_system is None:
        return Response(status_code=503, content="Vision system not initialized")

    latent = shared_state.latent_space
    reconstructed = shared_state.vision_system.vision_model.decode_frames([latent])[0]
    return numpy_array_to_response_image(reconstructed)


class DriveCommand(BaseModel):
    drive: float
    steer: float

@app.post("/drive")
def post_drive_command(command: DriveCommand, shared_state: SharedState = Depends(shared_state)):
    with SHARED_STATE_MUTEX:
        SHARED_STATE.action_tensor = ActionTensor(command.drive, command.steer)
    return Response(status_code=204)




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


def launch_webserver():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)

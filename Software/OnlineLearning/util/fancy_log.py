# pyright: strict
import functools
from typing import Any
import cv2
from ai.vision.image_util import NumpyImage


@functools.lru_cache
def _open_csv(filename: str, headings: list[str]):
    f = open(filename, "w")
    f.write(",".join(headings) + "\n")
    return f


def csv_saver(filename: str, data: dict[str, Any]):
    f = _open_csv(filename, tuple(sorted(list(data.keys()))))
    f.write(",".join(str(data[k]) for k in data.keys()) + "\n")
    f.flush()


@functools.lru_cache
def videowriter(
    filename: str,
    resolution: tuple[int, int],
    framerate: float,
) -> cv2.VideoWriter:
    return cv2.VideoWriter(
        filename,
        cv2.VideoWriter_fourcc(*"MPEG"),  # type: ignore
        framerate,
        resolution,
    )


def save_frame(filename: str, framerate: float, frame: NumpyImage):
    resolution = (frame.shape[1], frame.shape[0])
    writer = videowriter(filename, resolution, framerate)
    writer.write(frame)

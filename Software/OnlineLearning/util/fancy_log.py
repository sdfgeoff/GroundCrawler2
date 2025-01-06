import functools
import numpy
import cv2

@functools.lru_cache
def _open_csv(filename, headings):
    f = open(filename, "w")
    f.write(",".join(headings) + "\n")
    return f


def csv_saver(filename, data):
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
        cv2.VideoWriter_fourcc(*'MPEG'),
        framerate,
        resolution
    )
    

def save_frame(filename: str, framerate: float, frame: numpy.ndarray):
    resolution = (frame.shape[1], frame.shape[0])
    writer = videowriter(
        filename, resolution, framerate
    )
    writer.write(frame)

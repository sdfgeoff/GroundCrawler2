# pyright: strict
import cv2
from ai.vision.image_util import NumpyImage


class VideoFolderSource:
    def __init__(self, files: list[str | int]):
        self.files: list[str | int] = files
        self.file_id: int = 0
        self.cam: cv2.VideoCapture = cv2.VideoCapture(self.files[self.file_id])

        self._frame_fullres: NumpyImage | None = None

    def next_video(self) -> None:
        self.file_id += 1
        if self.file_id > len(self.files) - 1:
            self.file_id = 0
        # print(f"Using Filename: {self.files[self.file_id]}")
        self.cam = cv2.VideoCapture(self.files[self.file_id])

    def step(self) -> None:
        """Sample an image from the video source"""
        # reading from frame
        frame = self._get_frame()
        while frame is None:
            self.next_video()
            frame = self._get_frame()

        self._frame_fullres = frame

    def _get_frame(self) -> NumpyImage | None:
        ret, frame = self.cam.read()
        if ret:
            return frame
        return None

    def get_frame(self) -> NumpyImage:
        """Returns the entire image"""
        assert self._frame_fullres is not None

        return self._frame_fullres

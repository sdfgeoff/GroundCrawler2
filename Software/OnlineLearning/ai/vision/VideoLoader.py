import cv2
import numpy
import random


class VideoFolderSource():
    def __init__(self, files: list[str]):
        self.files: list[str] = files
        self.file_id: int = 0
        self.cam: cv2.VideoCapture = cv2.VideoCapture(self.files[self.file_id]) 

        self._frame_fullres: numpy.ndarray | None = None

    def next_video(self) -> None:
        self.file_id += 1
        if self.file_id > len(self.files) - 1:
            self.file_id = 0
        # print(f"Using Filename: {self.files[self.file_id]}")
        self.cam = cv2.VideoCapture(self.files[self.file_id]) 

    def step(self) -> None:
        """ Sample an image from the video source """
        # reading from frame 
        frame = self._get_frame()
        while frame is None:
            self.next_video()
            frame = self._get_frame()

        self._frame_fullres = frame
    
    def _get_frame(self) -> numpy.ndarray | None:
        ret,frame = self.cam.read() 
        if ret:
            return frame
        return None
    
    def get_full(self) -> numpy.ndarray:
        """ Returns the entire image """
        return self._frame_fullres



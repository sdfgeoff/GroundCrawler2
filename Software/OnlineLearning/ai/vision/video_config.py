from dataclasses import dataclass
import os
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_SIZE = 512
BATCH_SIZE = CACHE_SIZE
TRAINING_SET_MIN_FRAMES_BETWEEN_UPDATE = 1
AI_RESOLUTION = (160, 90)


VIDEO_SOURCES = [
    # "http://192.168.18.19/capture"  # ESP32
    0  # Webcam connected to the training PC
    # os.path.join("TestVideos", f) for f in os.listdir("TestVideos") if os.path.exists(os.path.join("TestVideos", f))
]

VISION_MODEL_PATH = os.path.normpath(os.path.join(os.path.abspath(__file__), "../model-1100000.raw"))

ENABLE_TRAIN = False
SAVE_TRAINING_VIDEO = True
SAVE_TRAINING_DATASET_VIDEO = False




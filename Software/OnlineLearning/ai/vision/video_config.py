from dataclasses import dataclass


@dataclass
class VideoModelConfig:
    initial_model_path: str | None
    device: str
    cache_size: int
    batch_size: int
    resolution: tuple[int, int]


@dataclass
class VideoSystemConfig:
    vision_model: VideoModelConfig

    train_enabled: bool
    training_video_show: bool
    training_video_save: bool

    training_dataset_video_save: bool
    training_dataset_frames_between_update: int

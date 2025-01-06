from typing import NamedTuple
import torch
from torch import nn
import numpy
from ai.vision import video_config
from ai.vision import train
from ai.vision.EncoderDecoderModel import EncoderDecoderModel
from ai.vision.VideoLoader import get_random_patch
from ai.vision.image_util import image_to_tensor, tensor_to_image, tile_images
from ai.vision.train import DataSet, memory_from_dataset, replace_random_image
from util import fancy_log

class VisionModel:
    def __init__(self, model_path: str | None = None):

        self.model = EncoderDecoderModel(
            (3, video_config.AI_RESOLUTION[1], video_config.AI_RESOLUTION[0])
        ).to(video_config.DEVICE)

        if model_path:
            self.model.load_state_dict(torch.load(
                model_path, 
                weights_only=True,
                map_location=torch.device(video_config.DEVICE)
            ))

        
        self.optimizer = None
        self.loss_fn = None
        self.training_data = None
        self.training_iteration = None


    def add_training_frame(self, frame: numpy.ndarray):
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        if self.loss_fn is None:
            nn.MSELoss()
        if self.training_iteration is None:
            self.training_iteration = 0

        if self.training_data is None:
            # Create lots of random patches
            initial_memory_images = [get_random_patch(frame, video_config.AI_RESOLUTION) for _ in range(video_config.CACHE_SIZE)]
            self.training_data = memory_from_dataset(DataSet(
                initial_memory_images
            ))
        else:
            self.training_data = replace_random_image(self.training_data, get_random_patch(frame, video_config.AI_RESOLUTION))


        if video_config.SAVE_TRAINING_DATASET_VIDEO:
            fancy_log.save_frame(
                "training_set2.avi",
                30 / video_config.TRAINING_SET_MIN_FRAMES_BETWEEN_UPDATE,
                tile_images([tensor_to_image(d) for d, _f in self.trainnig_data.dataset]).astype(numpy.uint8)
            )


    def do_training_pass(self):
        assert self.training_data is not None
        assert self.optimizer is not None
        assert self.loss_fn is not None
        assert self.training_iteration is not None

        training_loss = train.train(
            self.training_iteration, 
            self.training_data.dataloader, 
            self.model, 
            self.loss_fn, 
            self.optimizer
        )

        if self.training_iteration % 20000 == 0:
            torch.save(self.model.state_dict(), f"video_model-{self.training_iteration}.raw")


    def encode_decode_frames(self, frames: list[numpy.ndarray]) -> list[numpy.ndarray]:
        with torch.no_grad():
            self.model.eval()
            images_to_process = torch.stack(tuple(
                image_to_tensor(f) for f in frames
            ))

            result_gpu = self.model(images_to_process)
            return [tensor_to_image(r) for r in result_gpu]
    
    def encode_frames(self, frames: list[numpy.ndarray]) -> list[torch.Tensor]:
        with torch.no_grad():
            self.model.eval()
            images_to_process = torch.stack(tuple(
                image_to_tensor(f) for f in frames
            ))

            return self.model.do_encode(images_to_process)
        

    def decode_frames(self, frames: list[torch.Tensor]) -> list[numpy.ndarray]:
        with torch.no_grad():
            self.model.eval()
            images_to_process = torch.stack(frames)

            result_gpu = self.model.do_decode(images_to_process)
            return [tensor_to_image(r) for r in result_gpu]
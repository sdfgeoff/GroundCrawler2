from typing import NamedTuple
import cv2
import torch
from torch import nn
import numpy
from ai.vision import video_config
from ai.vision.DynamicMemory import DataSet, DynamicMemory, memory_from_dataset, replace_random_image
from ai.vision.EncoderDecoderModel import EncoderDecoderModel
from ai.vision.image_util import get_filtered_scaled, image_to_tensor, tensor_to_image, tile_images, get_filtered_patch
from util import fancy_log



class TrainingSetup(NamedTuple):
    optimizer: torch.optim.Optimizer
    loss_fn: nn.Module
    training_data: DynamicMemory


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

        self.training_setup = None
        self.training_iteration = 0


    def add_training_frame(self, frame: numpy.ndarray):
        if self.training_setup is None:
            initial_memory_images = [image_to_tensor(get_filtered_patch(frame, video_config.AI_RESOLUTION)) for _ in range(video_config.CACHE_SIZE)]
            training_data = memory_from_dataset(DataSet(
                initial_memory_images
            ))
            self.training_setup = TrainingSetup(
                optimizer=torch.optim.Adam(self.model.parameters(), lr=1e-5),
                loss_fn=nn.MSELoss(),
                training_data=training_data,
            )
        else:
            self.training_setup = TrainingSetup(
                optimizer=self.training_setup.optimizer,
                loss_fn=self.training_setup.loss_fn,
                training_data=replace_random_image(self.training_setup.training_data, get_filtered_patch(frame, video_config.AI_RESOLUTION)),
            )

        if video_config.SAVE_TRAINING_DATASET_VIDEO:
            fancy_log.save_frame(
                "training_set2.avi",
                30 / video_config.TRAINING_SET_MIN_FRAMES_BETWEEN_UPDATE,
                tile_images([tensor_to_image(d) for d, _f in self.training_setup.training_data.dataset]).astype(numpy.uint8)
            )


    def do_training_pass(self):
        self.training_iteration += 1
        self.model.train()
        for batch, (X, y) in enumerate(self.training_setup.training_data.dataloader):
            if self.training_iteration % (video_config.CACHE_SIZE / video_config.BATCH_SIZE) != batch:
                continue
    
            # Compute prediction error
            pred = self.model(X)
            loss = self.training_setup.loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            self.training_setup.optimizer.step()
            self.training_setup.optimizer.zero_grad()

        if self.training_iteration % 20000 == 0:
            torch.save(self.model.state_dict(), f"video_model-{self.training_iteration}.raw")

        return loss


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
        



class VisionSystem:
    def __init__(self):
        self.frame_counter = 0
        self.vision_model = VisionModel(
            model_path=video_config.VISION_MODEL_PATH
        )


    def ingest_frame(self, new_image_raw: numpy.ndarray):
        self.frame_counter += 1

        if video_config.ENABLE_TRAIN:
            if self.frame_counter % video_config.TRAINING_SET_MIN_FRAMES_BETWEEN_UPDATE == 0:
                self.vision_model.add_training_frame(new_image_raw)
            loss = self.vision_model.do_training_pass()

            fancy_log.csv_saver("vision_system_stats.csv", {
                "frame": self.frame_counter,
                "training_loss": loss,
            })

        new_image_scaled = get_filtered_scaled(new_image_raw, video_config.AI_RESOLUTION)
        latent_space = self.vision_model.encode_frames([new_image_scaled])[0]

        
        if video_config.SAVE_TRAINING_VIDEO:
            result = self.vision_model.encode_decode_frames([new_image_scaled])
            tile = tile_images([
                new_image_scaled,
                result[0],
            ]).astype(numpy.uint8)

            fancy_log.save_frame(
                "output2.avi",
                30,
                tile,
            )
            cv2.imshow('live', tile)
            cv2.waitKey(1)

        return latent_space
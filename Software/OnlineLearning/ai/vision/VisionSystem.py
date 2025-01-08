# pyright: strict
from typing import NamedTuple
import cv2
import torch
from torch import nn
import numpy
from ai.vision import video_config
from ai.vision.DynamicMemory import (
    DataSet,
    DynamicMemory,
    memory_from_dataset,
    replace_random_image,
)
from ai.vision.EncoderDecoderModel import EncoderDecoderModel
from ai.vision.image_util import (
    get_filtered_scaled,
    image_to_tensor,
    tensor_to_image,
    tile_images,
    get_filtered_patch,
    NumpyImage,
)
from util import fancy_log


class TrainingSetup(NamedTuple):
    optimizer: torch.optim.Optimizer
    loss_fn: nn.Module
    training_data: DynamicMemory


class VisionModel:
    def __init__(self, config: video_config.VideoModelConfig):
        self.model = EncoderDecoderModel(
            (3, config.resolution[1], config.resolution[0])
        ).to(config.device)

        if config.initial_model_path:
            self.model.load_state_dict(
                torch.load(  # type: ignore
                    config.initial_model_path,
                    weights_only=True,
                    map_location=torch.device(config.device),
                )
            )

        self.training_setup: TrainingSetup | None = None
        self.training_iteration = 0

        self.config = config

    def add_training_frame(self, frame: NumpyImage):
        if self.training_setup is None:
            initial_memory_images = [
                image_to_tensor(
                    get_filtered_patch(frame, self.config.resolution),
                    self.config.device,
                )
                for _ in range(self.config.cache_size)
            ]
            training_data = memory_from_dataset(
                DataSet(initial_memory_images),
                self.config.batch_size,
                self.config.device,
            )
            self.training_setup = TrainingSetup(
                optimizer=torch.optim.Adam(self.model.parameters(), lr=1e-5),
                loss_fn=nn.MSELoss(),
                training_data=training_data,
            )
        else:
            self.training_setup = TrainingSetup(
                optimizer=self.training_setup.optimizer,
                loss_fn=self.training_setup.loss_fn,
                training_data=replace_random_image(
                    self.training_setup.training_data,
                    get_filtered_patch(frame, self.config.resolution),
                ),
            )

    def do_training_pass(self):
        self.training_iteration += 1
        self.model.train()
        assert self.training_setup is not None

        loss = -1
        for batch, images in enumerate(self.training_setup.training_data.dataloader):
            if (
                self.training_iteration
                % (self.config.cache_size / self.config.batch_size)
                != batch
            ):
                continue

            # Compute prediction error
            pred = self.model(images)
            loss = self.training_setup.loss_fn(pred, images)

            # Backpropagation
            loss.backward()
            self.training_setup.optimizer.step()
            self.training_setup.optimizer.zero_grad()

        if self.training_iteration % 20000 == 0:
            torch.save(  # type: ignore
                self.model.state_dict(), f"video_model-{self.training_iteration}.raw"
            ) 

        return loss

    def encode_decode_frames(self, frames: list[NumpyImage]) -> list[NumpyImage]:
        with torch.no_grad():
            self.model.eval()
            images_to_process = torch.stack(
                tuple(image_to_tensor(f, device=self.config.device) for f in frames)
            )

            result_gpu = self.model(images_to_process)
            return [tensor_to_image(r) for r in result_gpu]

    def encode_frames(self, frames: list[NumpyImage]) -> list[torch.Tensor]:
        with torch.no_grad():
            self.model.eval()
            images_to_process = torch.stack(
                tuple(image_to_tensor(f, device=self.config.device) for f in frames)
            )

            return [t for t in self.model.do_encode(images_to_process)]

    def decode_frames(self, frames: list[torch.Tensor]) -> list[NumpyImage]:
        with torch.no_grad():
            self.model.eval()
            images_to_process = torch.stack(frames)  # type: ignore

            result_gpu = self.model.do_decode(images_to_process)
            return [tensor_to_image(r) for r in result_gpu]


class VisionSystem:
    def __init__(self, config: video_config.VideoSystemConfig):
        self.frame_counter = 0
        self.vision_model = VisionModel(config.vision_model)

        self.config = config

    def ingest_frame(self, new_image_raw: NumpyImage):
        self.frame_counter += 1

        if self.config.train_enabled:
            if (
                self.frame_counter % self.config.training_dataset_frames_between_update
                == 0
            ):
                self.vision_model.add_training_frame(new_image_raw)

                if self.config.training_dataset_video_save:
                    assert self.vision_model.training_setup is not None
                    fancy_log.save_frame(
                        "training_set2.avi",
                        30 / self.config.training_dataset_frames_between_update,
                        tile_images(
                            [
                                tensor_to_image(d)
                                for d, _f in self.vision_model.training_setup.training_data.dataset
                            ]
                        ).astype(numpy.uint8),
                    )

            loss = self.vision_model.do_training_pass()

            fancy_log.csv_saver(
                "vision_system_stats.csv",
                {
                    "frame": self.frame_counter,
                    "training_loss": loss,
                },
            )

        new_image_scaled = get_filtered_scaled(
            new_image_raw, self.config.vision_model.resolution
        )
        latent_space = self.vision_model.encode_frames([new_image_scaled])[0]

        if self.config.training_video_save or self.config.training_video_show:
            result = self.vision_model.encode_decode_frames([new_image_scaled])
            tile = tile_images(
                [
                    new_image_scaled,
                    result[0],
                ]
            ).astype(numpy.uint8)

            if self.config.training_video_save:
                fancy_log.save_frame(
                    "output2.avi",
                    30,
                    tile,
                )

            if self.config.training_video_show:
                cv2.imshow("live", tile)
                cv2.waitKey(1)

        return latent_space

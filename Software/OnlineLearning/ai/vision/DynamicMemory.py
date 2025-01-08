# pyright: strict

import random
from typing import NamedTuple
import torch
from torch.utils.data import DataLoader
from ai.vision.image_util import image_to_tensor, NumpyImage


class DataSet(torch.utils.data.Dataset[torch.FloatTensor]):
    def __init__(self, images: list[torch.FloatTensor]):
        self.images = images

    def __getitem__(self, idx: int) -> torch.FloatTensor:
        img = self.images[idx]
        return img

    def __len__(self):
        return len(self.images)


class DynamicMemory(NamedTuple):
    dataset: DataSet
    dataloader: DataLoader[torch.FloatTensor]
    device: str


def replace_random_image(memory: DynamicMemory, new_frame: NumpyImage) -> DynamicMemory:
    new_patch = image_to_tensor(new_frame, memory.device)
    # The idea is that replacing a random image gives the image memory an exponential falloff
    # and stops recent frames from dominating completely by ensuring some old frames still exist.
    # If it were a FIFO, then a 512 image buffer is ~17 seconds.
    # As an exponential falloff the oldest image will generally be much older.
    replace_frame_id = random.randrange(0, len(memory.dataset))

    new_dataset = [
        t for i, t in enumerate(memory.dataset.images) if i != replace_frame_id
    ] + [new_patch]
    memory = memory_from_dataset(
        DataSet(new_dataset), memory.dataloader.batch_size, memory.device
    )
    return memory


def memory_from_dataset(
    dataset: DataSet, batch_size: int | None, device: str
) -> DynamicMemory:
    return DynamicMemory(
        dataset, DataLoader(dataset, batch_size=batch_size), device=device
    )

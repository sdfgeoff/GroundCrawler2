
import random
from typing import NamedTuple
import numpy
import torch
from torch.utils.data import DataLoader
from ai.vision import video_config
from ai.vision.image_util import image_to_tensor


class DataSet(torch.utils.data.Dataset):
    def __init__(self, images: list[torch.FloatTensor]):
        self.images = images

    def __getitem__(self, idx):
        img = self.images[idx]
        return img, img
    
    def __len__(self):
        return len(self.images)



class DynamicMemory(NamedTuple):
    dataset: DataSet
    dataloader: DataLoader


def replace_random_image(memory: DynamicMemory, new_frame: numpy.ndarray) -> DynamicMemory:
    new_patch = image_to_tensor(new_frame)
    # The idea is that replacing a random image gives the image memory an exponential falloff
    # and stops recent frames from dominating completely by ensuring some old frames still exist.
    # If it were a FIFO, then a 512 image buffer is ~17 seconds.
    # As an exponential falloff the oldest image will generally be much older.
    replace_frame_id = random.randrange(0, len(memory.dataset))

    new_dataset = [
        t for i, (t, _f) in enumerate(memory.dataset) if i != replace_frame_id
    ] + [new_patch]
    assert len(new_dataset) == video_config.CACHE_SIZE
    memory = memory_from_dataset(DataSet(new_dataset))
    return memory


def memory_from_dataset(dataset: DataSet) -> DynamicMemory:
    return DynamicMemory(dataset, DataLoader(dataset, batch_size=video_config.BATCH_SIZE))


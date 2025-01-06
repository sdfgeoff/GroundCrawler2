import random
from typing import NamedTuple
from torch.utils.data import DataLoader
import torch
import numpy
from ai.vision.EncoderDecoderModel import EncoderDecoderModel
from ai.vision import video_config
from ai.vision.image_util import image_to_tensor
from ai.vision.VideoLoader import VideoFolderSource


class DataSet(torch.utils.data.Dataset):
    def __init__(self, images: list[torch.FloatTensor]):
        self.images = images

    def __getitem__(self, idx):
        img = self.images[idx]
        return img, img
    
    def __len__(self):
        return len(self.images)


def train(t, dataloader: DataLoader, model: EncoderDecoderModel, loss_fn, optimizer):
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        if t % (video_config.CACHE_SIZE / video_config.BATCH_SIZE) != batch:
            continue
  
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    return loss


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


def create_initial_dataset(video_source: VideoFolderSource) -> DynamicMemory:
    print("Creating initial dataset")
    initial_memory_images = []
    for f in range(video_config.CACHE_SIZE):
        #for _ in range(video_config.TRAINING_SET_MIN_FRAMES_BETWEEN_UPDATE):
        #    video_source.step()
        
        print(f"{f}/{video_config.CACHE_SIZE}", end="\r")
        # video_source.step()
        initial_memory_images.append(image_to_tensor(video_source.get_patch()))

    memory = memory_from_dataset(DataSet(
        initial_memory_images
    ))
    print("Initial Dataset Ready")
    return memory





import itertools
import math
from typing import Any
import numpy
import torch
import torchvision.transforms as transforms
from ai.vision import video_config
from PIL import Image
import functools
import cv2


def tensor_to_image(tensor: torch.Tensor) -> numpy.ndarray:
    #arr =  (tensor.permute(1,2,0)).clip(0, 1).detach().cpu().numpy().astype(numpy.float32)
    #return (arr * 255).astype(numpy.uint8)
    return numpy.array(transforms.functional.to_pil_image((tensor * 255.0).clip(0, 255).byte(), mode='RGB'))


def image_to_tensor(image: numpy.ndarray) -> torch.Tensor:
    #arr = image / 255
    #return torch.FloatTensor(arr).permute(2,0,1).to(config.DEVICE)
    return (transforms.functional.pil_to_tensor(Image.fromarray(image)).float() / 255).to(video_config.DEVICE)


def tile_images(images: list[numpy.ndarray[tuple[int, int, int], numpy.dtype[Any]]]) -> numpy.ndarray[tuple[int, int, int], numpy.dtype[Any]]:
    dimension = math.ceil(len(images) ** 0.5)
    height, width, channels = images[0].shape
    imgmatrix = numpy.zeros((dimension * height, dimension * width, channels))
    imgmatrix.fill(255)

    #Prepare an iterable with the right dimensions
    positions = itertools.product(range(dimension), range(dimension))

    for (y_i, x_i), img in zip(positions, images):
        x = x_i * width
        y = y_i * height
        imgmatrix[y:y+height, x:x+width, :] = img
    
    return imgmatrix


def upscale_image(image: numpy.ndarray, multiplier: int) -> numpy.ndarray:
    return image.repeat(multiplier,axis=0).repeat(multiplier,axis=1)

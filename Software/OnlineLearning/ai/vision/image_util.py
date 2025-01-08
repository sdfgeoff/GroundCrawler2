import itertools
import math
from typing import Any
import numpy
import torch
import torchvision.transforms as transforms  # type: ignore
from PIL import Image
import random
import cv2


NumpyImage = numpy.ndarray[tuple[int, int, int], numpy.dtype[Any]]


def tensor_to_image(tensor: torch.Tensor) -> NumpyImage:
    as_bytes = (tensor * 255.0).clip(0, 255).byte()
    as_image: Image = transforms.functional.to_pil_image(as_bytes, mode="RGB")  # type: ignore
    return numpy.array(as_image)  # type: ignore


def image_to_tensor(image: NumpyImage, device: str) -> torch.FloatTensor:
    as_image = Image.fromarray(image)
    as_tensor: torch.Tensor = transforms.functional.pil_to_tensor(as_image)  # type: ignore
    return (as_tensor.float() / 255.0).to(device)  # type: ignore


def tile_images(images: list[NumpyImage]) -> NumpyImage:
    dimension = math.ceil(len(images) ** 0.5)
    height, width, channels = images[0].shape
    imgmatrix = numpy.zeros((dimension * height, dimension * width, channels))
    imgmatrix.fill(255)

    # Prepare an iterable with the right dimensions
    positions = itertools.product(range(dimension), range(dimension))

    for (y_i, x_i), img in zip(positions, images):
        x = x_i * width
        y = y_i * height
        imgmatrix[y : y + height, x : x + width, :] = img

    return imgmatrix


def upscale_image(image: NumpyImage, multiplier: int) -> NumpyImage:
    return image.repeat(multiplier, axis=0).repeat(multiplier, axis=1)


def get_filtered_scaled(image: NumpyImage, resolution: tuple[int, int]) -> NumpyImage:
    """Returns the entire image scaled to resolution"""
    image = cv2.resize(image, resolution, interpolation=cv2.INTER_LINEAR)

    image = cv2.medianBlur(image, 3)
    return image


def get_filtered_patch(image: NumpyImage, resolution: tuple[int, int]) -> NumpyImage:
    """Returns a random part of the image at resolution"""
    image = get_random_patch(image, resolution)
    image = cv2.medianBlur(image, 3)
    return image


def get_random_patch(image: NumpyImage, target_resolution: tuple[int, int]):
    """
    Extract a random patch from an image with random scale, rotation, and position,
    and optionally scale the output to a target resolution.

    Parameters:
        image (np.ndarray): Input image.
        min_scale (float): Minimum scale factor for the patch size relative to the image size.
        max_scale (float): Maximum scale factor for the patch size relative to the image size.
        target_resolution (tuple): Target resolution (width, height) for the output patch. If None, no scaling is applied.

    Returns:
        np.ndarray: Extracted patch as a new image.
    """
    height, width = image.shape[:2]

    # Random scale factor
    scale = random.uniform(0.1, 0.6)
    patch_width = int(scale * width)
    patch_height = int(scale * height)

    # Ensure patch dimensions are at least 1x1
    patch_width = max(1, patch_width)
    patch_height = max(1, patch_height)

    # Random rotation angle
    angle = random.uniform(0, 360)

    # Random center position for the patch
    center_x = random.randint(patch_width // 2, width - patch_width // 2)
    center_y = random.randint(patch_height // 2, height - patch_height // 2)

    # Define the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)

    # Apply the rotation to the image
    rotated_image = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    # Extract the patch from the rotated image
    top_left_x = center_x - patch_width // 2
    top_left_y = center_y - patch_height // 2
    bottom_right_x = top_left_x + patch_width
    bottom_right_y = top_left_y + patch_height

    # Ensure patch coordinates are within the image bounds
    top_left_x = max(0, top_left_x)
    top_left_y = max(0, top_left_y)
    bottom_right_x = min(width, bottom_right_x)
    bottom_right_y = min(height, bottom_right_y)

    patch = rotated_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    # Scale the patch to the target resolution if specified
    patch = cv2.resize(patch, target_resolution, interpolation=cv2.INTER_LINEAR)

    return patch

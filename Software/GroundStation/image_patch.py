import cv2
import numpy as np
import random

def extract_random_patch(image, min_scale=0.1, max_scale=0.5, target_resolution=None):
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
    scale = random.uniform(min_scale, max_scale)
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
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

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
    if target_resolution is not None:
        patch = cv2.resize(patch, target_resolution, interpolation=cv2.INTER_LINEAR)

    return patch


# Example usage
if __name__ == "__main__":
    # Load an example image
    image = cv2.imread("snap.png")
    assert image is not None
    
    # Display the patch
    while True:
        random_patch = extract_random_patch(image, target_resolution=(360, 240))
        cv2.imshow("Random Patch", random_patch)
        cv2.waitKey(1000)

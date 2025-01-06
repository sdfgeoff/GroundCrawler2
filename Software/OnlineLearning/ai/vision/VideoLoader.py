import cv2
import numpy
import random


class VideoFolderSource():
    def __init__(self, files: list[str], resolution: tuple[int, int]):
        self.files: list[str] = files
        self.file_id: int = 0
        self.cam: cv2.VideoCapture = cv2.VideoCapture(self.files[self.file_id]) 
        self.resolution = resolution

        self._frame_fullres: numpy.ndarray | None = None

    def next_video(self) -> None:
        self.file_id += 1
        if self.file_id > len(self.files) - 1:
            self.file_id = 0
        # print(f"Using Filename: {self.files[self.file_id]}")
        self.cam = cv2.VideoCapture(self.files[self.file_id]) 

    def step(self) -> None:
        """ Sample an image from the video source """
        # reading from frame 
        frame = self._get_frame()
        while frame is None:
            self.next_video()
            frame = self._get_frame()

        self._frame_fullres = frame
    
    def _get_frame(self) -> numpy.ndarray | None:
        ret,frame = self.cam.read() 
        if ret:
            return frame
        return None
    
    def get_full(self) -> numpy.ndarray:
        """ Returns the entire image """
        return self._frame_fullres
        
    def get_scaled(self) -> numpy.ndarray:
        """ Returns the entire image scaled to resolution """
        image = cv2.resize(
            self._frame_fullres,
            self.resolution, 
            interpolation = cv2.INTER_LINEAR)
        
        image = cv2.medianBlur(image,3)
        return image
    
    def get_patch(self) -> numpy.ndarray:
        """ Returns a random part of the image at resolution """
        assert self._frame_fullres is not None
        image = self._frame_fullres
        image = get_random_patch(image, self.resolution)
        image = cv2.medianBlur(image,3)
        assert image is not None
        return image




def get_random_patch(image, target_resolution):
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




# def get_random_patch(frame_fullres: numpy.ndarray, resolution: tuple[int, int]):
#     scale = random.uniform(0.1, 0.6)
#     center = [
#         random.uniform(0.0, 1.0),
#         random.uniform(0.0, 1.0),
#     ]
#     flipx = random.getrandbits(1)
#     flipy = random.getrandbits(1)
#     angle = random.uniform(0.0, 360.0)



#     size = (int(frame_fullres.shape[0] * scale), int(frame_fullres.shape[0] * scale))
#     center = [
#         int(center[0] * (frame_fullres.shape[0] - size[0])),
#         int(center[0] * (frame_fullres.shape[1] - size[1]))
#     ]

#     res = frame_fullres[
#         center[0]:center[0]+size[0],
#         center[1]:center[1]+size[1]
#     ]

#     scaled = cv2.resize(
#         res,
#         resolution, 
#         interpolation = cv2.INTER_LINEAR)
    
#     flipped = scaled
#     if flipx:
#         flipped = numpy.flip(flipped, 0)
#     if flipy:
#         flipped = numpy.flip(flipped, 1) 

#     return flipped


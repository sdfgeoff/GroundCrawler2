from typing import Any, NamedTuple
import torch
import functools
from torch import nn
import numpy
from ai.vision.EncoderDecoderModel import EncoderDecoderModel
from ai.vision.VideoLoader import VideoFolderSource
from ai.vision.image_util import tile_images, image_to_tensor, tensor_to_image, upscale_image
import time
from ai.vision.VisionSystem import VisionModel
from ai.vision import video_config
import cv2

from util import fancy_log
import os




def main():
    vid = VideoFolderSource(video_config.VIDEO_SOURCES, video_config.AI_RESOLUTION)
    # test_vid = VideoFolderSource(["Data/VID_20241126_181927.mp4"], video_config.AI_RESOLUTION)


    vision_system = VisionModel(
        model_path=os.path.normpath(os.path.join(os.path.abspath(__file__), "../ai/vision/model-1100000.raw"))
    )

    vid.step()


    sliders = torch.zeros(121).float()
    cv2.namedWindow('live')

    def make_slider_callback(slider_id):
        def set_slider(x):
            sliders[slider_id] = x/10
        return set_slider

    for i in range(121):
        cv2.createTrackbar(str(i), 'live', -250, 250, make_slider_callback(i))

    t = 0
    prev_time = time.time()
    while True:
        t += 1
        vid.step()

        if video_config.ENABLE_TRAIN:
            if t % video_config.TRAINING_SET_MIN_FRAMES_BETWEEN_UPDATE == 0:
                new_image_raw = vid.get_full()
                vision_system.add_training_frame(new_image_raw)

            loss = vision_system.do_training_pass()
            

        explored_space = vision_system.decode_frames([sliders])
        

        new_image_scaled = vid.get_scaled()
        result = vision_system.encode_decode_frames([new_image_scaled])
        tile = tile_images([
            new_image_scaled,
            result[0],
            explored_space[0],
        ]).astype(numpy.uint8)


        # latent_space = vision_system.encode_frames([new_image_scaled])
        #print(latent_space[0])


        fancy_log.save_frame(
            "output2.avi",
            30,
            tile,
        )
            
        cv2.imshow('live', tile)
        cv2.waitKey(1)



if __name__ == "__main__":
    main()




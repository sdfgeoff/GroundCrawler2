from threading import Thread
import torch
from ai.vision.VisionSystem import VisionSystem
from ai.vision.VideoLoader import VideoFolderSource
import time
from ai.vision import video_config
from ai.vision.image_util import get_filtered_scaled


import chassis
from webserver.app import SHARED_STATE_MUTEX, SHARED_STATE, launch_webserver


vision_config = video_config.VideoSystemConfig(
    vision_model=video_config.VideoModelConfig(
        initial_model_path=None,  # 'ai/vision/model-1100000.raw',
        device="cuda" if torch.cuda.is_available() else "cpu",
        cache_size=512,
        batch_size=512,
        resolution=(160, 90),
    ),
    train_enabled=True,
    training_video_show=False,
    training_video_save=False,
    training_dataset_video_save=False,
    training_dataset_frames_between_update=1,
)


def ai_thread():
    # VID_FOLDER = "../../../pytorch-learning/Data"
    VIDEO_SOURCE: list[str | int] = [
        #     # "http://192.168.18.19/capture"  # ESP32
        0  # Webcam connected to the training PC
        #     os.path.join(VID_FOLDER, f) for f in os.listdir(VID_FOLDER) if os.path.exists(os.path.join(VID_FOLDER, f))
    ]
    vid = VideoFolderSource(VIDEO_SOURCE)
    vid.step()

    vision_system = VisionSystem(vision_config)

    t = 0
    prev_time = time.time()
    while True:
        t += 1
        vid.step()
        new_frame = vid.get_frame()

        with SHARED_STATE_MUTEX:
            shared_state = SHARED_STATE
            latent_space = vision_system.ingest_frame(new_frame)

            shared_state.camera_image = get_filtered_scaled(
                new_frame, vision_config.vision_model.resolution
            )
            shared_state.latent_space = latent_space
            shared_state.vision_system = vision_system

            chassis.drive(shared_state.action_tensor.forwards, shared_state.action_tensor.steer)

        # Print FPS
        if t % 10 == 0:
            new_time = time.time()
            print(f"FPS: {10 / (new_time - prev_time)}")
            prev_time = new_time


if __name__ == "__main__":
    # ai_thread()
    ai_thread_instance = Thread(target=ai_thread, daemon=True)
    ai_thread_instance.start()

    webserver_thread_instance = Thread(target=launch_webserver, daemon=True)
    webserver_thread_instance.start()

    while True:
        time.sleep(1.0)  # Everythign happens in threads

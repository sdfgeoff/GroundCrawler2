from ai.vision.VisionSystem import VisionSystem
from ai.vision.VideoLoader import VideoFolderSource
import time
from ai.vision import video_config



def main():
    vid = VideoFolderSource(video_config.VIDEO_SOURCES, video_config.AI_RESOLUTION)

    vision_system = VisionSystem()
    vid.step()

    t = 0
    prev_time = time.time()
    while True:
        t += 1
        vid.step()
        new_frame = vid.get_full()
        latent_space = vision_system.ingest_frame(new_frame)

        

 
        # Print FPS
        if t % 10 == 0:
            new_time = time.time()
            print(f"FPS: {10 / (new_time - prev_time)}")
            prev_time = new_time



if __name__ == "__main__":
    main()




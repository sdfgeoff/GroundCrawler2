import cv2
import functools
import numpy
import urllib.request
import asyncio
import time


ESP_IP = "192.168.18.19"
image_url = f"http://{ESP_IP}/capture"
drive_url = f"http://{ESP_IP}/drive?forward={{forward}}&steer={{steer}}"
configure_url = f"http://{ESP_IP}/config?framesize=5"  # Higher "quality" is higher compression....


@functools.lru_cache
def videowriter(
    filename: str,
    resolution: tuple[int, int],
    framerate: float,
) -> cv2.VideoWriter:
    return cv2.VideoWriter( 
        filename,
        cv2.VideoWriter_fourcc(*'MPEG'),
        framerate,
        resolution
    )
    

def save_frame(filename: str, framerate: float, frame: numpy.ndarray):
    resolution = (frame.shape[1], frame.shape[0])
    writer = videowriter(
        filename, resolution, framerate
    )
    writer.write(frame)


def fetch_image():
    cam = cv2.VideoCapture(image_url)
    ret, img = cam.read()
    return img
    

def drive(forwards, steer):
    url = drive_url.format(
        forward=int(forwards * 100),
        steer=int(steer * 100)
    )
    contents = urllib.request.urlopen(
        url
    ).read()
    


async def main():
    
    forwards = 0
    steer = 0
    
    i = 0
    
    
    contents = urllib.request.urlopen(
        configure_url
    ).read()
    print(contents)
    time.sleep(0.5)
    
    while True:
        i += 1
        
        img = fetch_image()
        
        if img is None:
            continue
            
        save_frame("recording.avi", 30, img)
        cv2.imshow("image", img)
        key = cv2.waitKey(1)
        
        forwards_target = 0
        steer_target = 0

        if key == 27:  # ESC
            break
        if key == 82:
            forwards_target = 1
        if key == 84:
            forwards_target = -1
        if key == 83:
            steer_target = 1
        if key == 81:
            steer_target = -1
        
            
        steer = steer * 0.9 + steer_target * 0.1
        forwards = forwards * 0.9 + forwards_target * 0.1
        
        if i % 2 == 0:
            drive(forwards, steer)
            
        


if __name__ == "__main__":
    asyncio.run(main())

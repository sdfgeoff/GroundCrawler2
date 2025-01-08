import cv2
import urllib.request


ESP_IP = "192.168.18.19"
image_url = f"http://{ESP_IP}/capture"
drive_url = f"http://{ESP_IP}/drive?forward={{forward}}&steer={{steer}}"
configure_url = (
    f"http://{ESP_IP}/config?framesize=5"  # Higher "quality" is higher compression....
)


def fetch_image():
    cam = cv2.VideoCapture(image_url)
    ret, img = cam.read()
    return img


def drive(forwards, steer):
    url = drive_url.format(forward=int(forwards * 100), steer=int(steer * 100))
    _contents = urllib.request.urlopen(url).read()

import cv2
import io
import socket
import struct
import time
import pickle
import zlib
import qi
from naoqi import ALProxy
import vision_definitions
from PIL import Image
import sys


ip = "10.0.0.83"
port = 9559
tts = ALProxy("ALTextToSpeech", "10.0.0.83", 9559)
tts.say("Connected")
session = qi.Session()
try:
    session.connect("tcp://" + ip + ":" + str(port))
except RuntimeError:
    print ("Can't connect to Naoqi at ip \"" + ip + "\" on port " + str(port) +".\n"
            "Please check your script arguments. Run with -h option for help.")
    sys.exit(1)

# HOST=socket.gethostbyname(socket.gethostname())
HOST = "10.0.0.85"

print(HOST)
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, 8485))
connection = client_socket.makefile('wb')

# cam = cv2.VideoCapture(0)
#
# cam.set(3, 320);
# cam.set(4, 240);

img_counter = 0
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

video_service = session.service("ALVideoDevice")

# Register a Generic Video Module
resolution = vision_definitions.kQVGA
colorSpace = vision_definitions.kRGBColorSpace
fps = 20

nameId = video_service.subscribe("python_GVM", resolution, colorSpace, fps)
print("Pepper camera set up...")

while True:
# for _ in range(1):
    # ret, frame = cam.read()

    naoImage = video_service.getImageRemote(nameId)
    # Get the image size and pixel array.
    imageWidth = naoImage[0]
    imageHeight = naoImage[1]
    array = naoImage[6]
    image_string = str(bytearray(array))

    # Create a PIL Image from our pixel array.
    img = Image.frombytes("RGB", (imageWidth, imageHeight), image_string)

    # Save the image.
    img.save(r"imagesFromPepper/liveStream.png", "PNG")

    frame = cv2.imread('imagesFromPepper/liveStream.png')
    try:
        result, frame = cv2.imencode('.jpg', frame, encode_param)
    except:
        continue
#    data = zlib.compress(pickle.dumps(frame, 0))
    data = pickle.dumps(frame, 0)
    size = len(data)


    print("{}: {}".format(img_counter, size))
    client_socket.sendall(struct.pack(">L", size) + data)
    img_counter += 1

# cam.release()
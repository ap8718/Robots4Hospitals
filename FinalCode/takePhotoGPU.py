import qi
import os
import sys
import time
import vision_definitions
# import cv2
import almath
from PIL import Image, ImageEnhance
import tablet

# Change this to match GPU machine IP and port
IP = "root@2.tcp.ngrok.io"
PORT = "18972"


def main(session):
    """
    This is just an example script that shows how images can be accessed
    through ALVideoDevice in Python.
    Nothing interesting is done with the images in this example.
    """

    # Get the service ALVideoDevice.
    video_service = session.service("ALVideoDevice")


    # Register a Generic Video Module
    resolution = vision_definitions.kVGA
    colorSpace = vision_definitions.kRGBColorSpace
    fps = 20

    nameId = video_service.subscribe("python_GVM", resolution, colorSpace, fps)
    tts = session.service("ALTextToSpeech")
    tts.say('Please stay still, I am going to take a picture of you')

    bap = session.service('ALBasicAwareness')

    motion_service  = session.service("ALMotion")

    print 'getting images in remote'
    bap.pauseAwareness()
    print "getting image 0"
    motion_service.setStiffnesses("Head", 1.0)
    names      = "Head"

    angleLists = [0*almath.TO_RAD,-7*almath.TO_RAD]

    motion_service.angleInterpolationWithSpeed(names, angleLists, 0.4)

    time.sleep(1)
    naoImage = video_service.getImageRemote(nameId)




    # Get the image size and pixel array.
    imageWidth = naoImage[0]
    imageHeight = naoImage[1]
    array = naoImage[6]
    image_string = str(bytearray(array))


    # Create a PIL Image from our pixel array.
    img = Image.frombytes("RGB", (imageWidth, imageHeight), image_string)


    # Save the image.
    img.save(r"imagesFromPepper/analysis0.png", "PNG")

    tts.say('Picture taken')
    tablet.main(session)
    bap.resumeAwareness()
    video_service.unsubscribe(nameId)

    os.system("scp -P " + PORT + " imagesFromPepper/analysis0.png " + IP + ":~/Robots4Hospitals/FinalCode/imagesFromPepper/")
    time.sleep(30)
    os.system("scp -r -P " + PORT + " " + IP + ":~/Robots4Hospitals/FinalCode/Results/ .")

    mask = open('Results/MaskText', 'r')
    visor = open('Results/VisorText', 'r')
    glove = open('Results/GloveText', 'r')
    gown = open('Results/GownText', 'r')


    tts.say(visor.read())
    time.sleep(0.5)
    tts.say(mask.read())
    print(mask.read())
    time.sleep(0.5)
    tts.say(glove.read())
    time.sleep(0.5)
    print(glove.read())
    tts.say(gown.read())

    print("All done!")



if __name__ == "__main__":

    ip = "10.0.0.83"
    port = 9559
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default=ip,
                        help="Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.")
    parser.add_argument("--port", type=int, default=port,
                        help="Naoqi port number")

    args = parser.parse_args()
    session = qi.Session()
    try:
        session.connect("tcp://" + ip + ":" + str(port))
    except RuntimeError:
        print ("Can't connect to Naoqi at ip \"" + ip + "\" on port " + str(port) +".\n"
               "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)
    main(session)

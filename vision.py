import naoqi
from naoqi import ALProxy
import qi
import sys
import time
import vision_definitions
from PIL import Image
import detetc_mask_video

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

    print 'getting images in remote'
    for i in range(0, 1):
        print "getting image " + str(i)
        naoImage = video_service.getImageRemote(nameId)
        # Get the image size and pixel array.
        imageWidth = naoImage[0]
        imageHeight = naoImage[1]
        array = naoImage[6]
        image_string = str(bytearray(array))

        # Create a PIL Image from our pixel array.
        im = Image.frombytes("RGB", (imageWidth, imageHeight), image_string)

        # Save the image.
        im.save(r"imagesFromPepper/camImage.png", "PNG")

        im.show()
        time.sleep(5)

    video_service.unsubscribe(nameId)

    

    (locs, preds) = detect_mask_video.detect_and_predict_mask(im)

    print((locs,preds))
    
    


if __name__ == "__main__":
   
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--ip", type=str, default="127.0.0.1",
    #                     help="Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.")
    # parser.add_argument("--port", type=int, default=9559,
    #                     help="Naoqi port number")

    # args = parser.parse_args()
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
    main(session)
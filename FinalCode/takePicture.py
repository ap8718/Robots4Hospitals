import naoqi
from naoqi import ALProxy
import qi
import sys
import time
import vision_definitions
import cv2
import almath
from PIL import Image, ImageEnhance
import tablet




def main(session):
    """
    This is just an example script that shows how images can be accessed
    through ALVideoDevice in Python.
    Nothing interesting is done with the images in this example.
    """
  
    # Get the service ALVideoDevice.

    #video_service = session.service("ALVideoDevice")

    video_service = session.service("ALVideoDevice")


    # Register a Generic Video Module
    # resolution = vision_definitions.k4VGA
    resolution = vision_definitions.kVGA
    colorSpace = vision_definitions.kRGBColorSpace
    fps = 20

    nameId = video_service.subscribe("python_GVM", resolution, colorSpace, fps)
    tts = ALProxy("ALTextToSpeech", "10.0.0.83", 9559)
    tts.say('Please stay still, I am going to take a picture of you')

    bap = ALProxy('ALBasicAwareness', '10.0.0.83', 9559)
    
    motion_service  = session.service("ALMotion")

    print 'getting images in remote'
    for i in range(0, 1):
        bap.pauseAwareness()
        print "getting image " + str(i)
        motion_service.setStiffnesses("Head", 1.0)
        names      = "Head"
        
        angleLists = [0*almath.TO_RAD,-7*almath.TO_RAD]

        motion_service.angleInterpolationWithSpeed(names, angleLists, 0.6)

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
        img.save(r"imagesFromPepper/analysis1.png", "PNG")

        

        # enhancer = ImageEnhance.Contrast(img)

        # factor = 0.6 #gives original image
        # im_output = enhancer.enhance(factor)
        # im_output.save(r'imagesFromPepper/lesscontrast.png')



    #result = merge_images(r"imagesFromPepper/camImage1.png", r"imagesFromPepper/camImage2.png")
    #result.save(r"imagesFromPepper/camImage.png", "PNG")
      
    tts.say('Picture taken')
    bap.resumeAwareness()
    tablet.main(session)
    video_service.unsubscribe(nameId)
  


    
    


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
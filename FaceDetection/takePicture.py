import naoqi
from naoqi import ALProxy
import qi
import sys
import time
import vision_definitions
from PIL import Image
import detect_mask_video
import cv2
#
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
    tts = ALProxy("ALTextToSpeech", "10.0.0.83", 9559)
    tts.say('Please stay still, I am going to take a picture of you')
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
        img = Image.frombytes("RGB", (imageWidth, imageHeight), image_string)

        # Save the image.
        img.save(r"imagesFromPepper/camImage.png", "PNG")

      

    video_service.unsubscribe(nameId)
    picture = cv2.imread(r"imagesFromPepper/camImage.png")
    
    (locs, preds) = detect_mask_video.detect_and_predict_mask(picture)
    print((locs, preds) )
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
        print(mask)
        # determine the class label and color we'll use to draw
        # the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
    
        tts.say( label + ' Detected')

        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(picture, label, (startX, startY - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(picture, (startX, startY), (endX, endY), color, 2)

	
	cv2.imshow("Frame", picture)

    while True:	
        key = cv2.waitKey(1) 

    
    


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
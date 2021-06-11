import qi
import os
import time
import takePhotoGPU as photo
import movement as move
import sys
import argparse
from datetime import datetime


def main(session):
    tts = session.service("ALTextToSpeech")
    face_detection = session.service("ALSpeechRecognition")
    tablet = session.service("ALTabletService")
    awareness = session.service("ALBasicAwareness")

    awareness.pauseAwareness()
    print(awareness.isAwarenessPaused())
    tts.say('Okay')

    t1 = datetime.now()
    move.main(session)
    # photo.main(session)
    t2 = datetime.now()

    tablet.hideImage()

    awareness.resumeAwareness()

if __name__ == "__main__":
   
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--ip", type=str, default="127.0.0.1",
    #                     help="Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.")
    # parser.add_argument("--port", type=int, default=9559,
    #                     help="Naoqi port number")

    # args = parser.parse_args()
    ip = "10.0.0.83"
    port = 9559
    session = qi.Session()
    try:
        session.connect("tcp://" + ip + ":" + str(port))
    except RuntimeError:
        print ("Can't connect to Naoqi at ip \"" + ip + "\" on port " + str(port) +".\n"
               "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)
    main(session)
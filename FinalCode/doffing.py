import naoqi
import time
import sys
import argparse
from datetime import datetime
import subprocess
import qi
from naoqi import ALProxy
import almath
import vision_definitions
import cv2
import almath
from PIL import Image, ImageEnhance
import tablet
import os

IP = "10.0.0.83"
PORT = 9559

    

def main(session):
    bap = ALProxy('ALBasicAwareness', '10.0.0.83', 9559)
    motion_service  = session.service("ALMotion")

    ms = session.service("ALMotion")
    tts = session.service("ALTextToSpeech")
    try:
        videoRecorderProxy = ALProxy("ALVideoRecorder", IP, PORT)
    except Exception, e:
        print "Error when creating ALVideoRecorder proxy:"
        print str(e)
        exit(1)



    
    tts.say('Commencing Doffing')

    tts.say('You now have the next 12 seconds to doff your gown. Please make sure not to touch the inside of your gown or any bare skin with your gloves on')
    

    # Motion for recording video
    bap.pauseAwareness()
    motion_service.setStiffnesses("Head", 1.0)
    names      = "Head"
    angleLists = [0*almath.TO_RAD,-7*almath.TO_RAD]
    motion_service.angleInterpolationWithSpeed(names, angleLists, 0.6)
    time.sleep(1)

    #recording video
    videoRecorderProxy.setFrameRate(15.0)
    videoRecorderProxy.startRecording("/home/nao/recordings/cameras", "test")
    print "Video record started."
    time.sleep(12) # Duration of video 
    videoInfo = videoRecorderProxy.stopRecording()
    tts.say("Video taken!")
    print "Video was saved on the robot: ", videoInfo[1]
    print "Total number of frames: ", videoInfo[0]

    #Sending video to GPU server
    os.system("scp nao@10.0.0.83:~/recordings/cameras/test.avi ./imagesFromPepper")
    os.system("scp -P 19563 imagesFromPepper/test.avi root@2.tcp.ngrok.io:/root/Robots4Hospitals/Gown_doff")

    time.sleep(20)

    os.system("scp -P 19563 root@2.tcp.ngrok.io:/root/Robots4Hospitals/Gown_doff/GownDoffingText .")
    f = open('GownDoffingText', 'r')
    tts.say(f.read())


    tts.say('Please now wash your inner gloves')

    tts.say('You now have the next 15 seconds to take of your visor, make sure to not touch the front of the visor at any point')

    tts.say('Please now remove your inner gloves')
    time.sleep(2)
    tts.say('You can now remove your mask, and leave the ward')
    bap.resumeAwareness()
    print("All done!")

if __name__ == "__main__":
    session = qi.Session()
    try:
        session.connect("tcp://10.0.0.83:9559")
    except RuntimeError:
        print ("Can't connect to Naoqi at ip '10.0.0.83' on port '9559'.\n"
               "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)
    main(session)
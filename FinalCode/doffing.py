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

GPU_IP = "root@4.tcp.ngrok.io"
GPU_PORT = "19969"


def main(session):

    gownSafe = False
    visorSafe = False
    glovesSafe = False

    bap = ALProxy('ALBasicAwareness', '10.0.0.83', 9559)
    motion_service  = session.service("ALMotion")

    ms = session.service("ALMotion")
    tts = session.service("ALTextToSpeech")
    try:
        videoRecorderProxy = ALProxy("ALVideoRecorder", '10.0.0.83', 9559)
    except Exception, e:
        print "Error when creating ALVideoRecorder proxy:"
        print str(e)
        exit(1)
    
    tts.say('Commencing Doffing')
    # tts.say('You now have the next 12 seconds to doff your gown and outer gloves. Please make sure not to touch the inside of your gown or any bare skin with your gloves on')
    tts.say('You now have the next 12 seconds to doff your gown and outer gloves.')

    # Motion for recording video
    bap.pauseAwareness()
    motion_service.setStiffnesses("Head", 1.0)
    names      = "Head"
    angleLists = [0*almath.TO_RAD,-7*almath.TO_RAD]
    motion_service.angleInterpolationWithSpeed(names, angleLists, 0.6)
    time.sleep(1)

    #GOWN DOFFING: Recording video
    videoRecorderProxy.setFrameRate(15.0)
    videoRecorderProxy.startRecording("/home/nao/recordings/cameras", "gown")
    print "Video record started."
    time.sleep(12) # Duration of video 
    videoInfo = videoRecorderProxy.stopRecording()
    tts.say("Video taken!")
    print "Video was saved on the robot: ", videoInfo[1]
    print "Total number of frames: ", videoInfo[0]
    #Sending video to GPU server
    os.system("scp nao@10.0.0.83:~/recordings/cameras/gown.avi ./imagesFromPepper")
    os.system("scp -P  " + GPU_PORT + " imagesFromPepper/gown.avi " + GPU_IP + ":/root/Robots4Hospitals/FinalCode/Gown_doff")
    time.sleep(20)
    os.system("scp -P  " + GPU_PORT + " " + GPU_IP + ":/root/Robots4Hospitals/FinalCode/Gown_doff/GownDoffingText .")
    f = open('Gown_doff/GownDoffingText', 'r')
    string = f.read()
    tts.say(string)
    if string != 'Contamination detected!':
        gownSafe = True
    time.sleep(1)

    tts.say('Please now wash your inner gloves')
    time.sleep(5)

    # VISOR DOFFING: Recording video
    # tts.say('You now have the next 5 seconds to doff your visor, make sure to not touch the front of the visor at any point')
    tts.say('You now have the next 5 seconds to doff your visor.')

    videoRecorderProxy.setFrameRate(15.0)
    videoRecorderProxy.startRecording("/home/nao/recordings/cameras", "visor")
    print "Video record started."
    time.sleep(5) # Duration of video 
    videoInfo = videoRecorderProxy.stopRecording()
    tts.say("Video taken!")
    print "Video was saved on the robot: ", videoInfo[1]
    print "Total number of frames: ", videoInfo[0]
    #Sending video to GPU server
    os.system("scp nao@10.0.0.83:~/recordings/cameras/visor.avi ./imagesFromPepper")
    os.system("scp -P  " + GPU_PORT + " imagesFromPepper/visor.avi  " + GPU_IP + ":/root/Robots4Hospitals/FinalCode/Visor_doff")
    time.sleep(20)
    os.system("scp -P  " + GPU_PORT + "  " + GPU_IP + ":/root/Robots4Hospitals/FinalCode/Visor_doff/VisorDoffingText .")
    f = open('Visor_doff/VisorDoffingText', 'r')
    string = f.read()
    tts.say(string)
    if string != 'Contamination detected!':
        visorSafe = True
    time.sleep(1)

    # GLOVE DOFFING: Recording video
    # tts.say('Please now remove your inner gloves')
    tts.say('You now have the next 10 seconds to doff your inner gloves')
    # time.sleep(2)

    videoRecorderProxy.setFrameRate(15.0)
    videoRecorderProxy.startRecording("/home/nao/recordings/cameras", "gloves")
    print "Video record started."
    time.sleep(10) # Duration of video 
    videoInfo = videoRecorderProxy.stopRecording()
    tts.say("Video taken!")
    print "Video was saved on the robot: ", videoInfo[1]
    print "Total number of frames: ", videoInfo[0]
    #Sending video to GPU server
    os.system("scp nao@10.0.0.83:~/recordings/cameras/gloves.avi ./imagesFromPepper")
    os.system("scp -P  " + GPU_PORT + " imagesFromPepper/gloves.avi  " + GPU_IP + ":/root/Robots4Hospitals/FinalCode/Glove_doff")
    time.sleep(20)
    os.system("scp -P  " + GPU_PORT + "  " + GPU_IP + ":/root/Robots4Hospitals/FinalCode/Glove_doff/GloveDoffingText .")
    f = open('Glove_doff/GloveDoffingText', 'r')
    string = f.read()
    tts.say(string)
    print(string)
    if string != "Contamination detected!":
        glovesSafe = True
    time.sleep(1)

    # print((gownSafe, visorSafe, glovesSafe))

    if gownSafe and visorSafe and glovesSafe:
        tts.say('You can now leave the ward and remove your mask')
    else:
        tts.say('Please seek assisstance to decontaminate yourself if you believe you have been contaminated')
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
# This test demonstrates how to use the ALVideoRecorder module.
# Note that you might not have this module depending on your distribution
import os
import sys
import time
from naoqi import ALProxy
import cv2
import almath

# Replace this with your robot's IP address
IP = "10.0.0.83"
PORT = 9559

def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite("recordings/image"+str(count)+".jpg", image)     # save frame as JPG file
    return hasFrames

# Create a proxy to ALVideoRecorder
try:
  videoRecorderProxy = ALProxy("ALVideoRecorder", IP, PORT)
except Exception, e:
  print "Error when creating ALVideoRecorder proxy:"
  print str(e)
  exit(1)

tts = ALProxy("ALTextToSpeech", IP, PORT)
motion_service = ALProxy("ALMotion", IP, PORT)
bas = ALProxy("ALBasicAwareness", IP, PORT)

time.sleep(2)
videoRecorderProxy.setFrameRate(15.0)
# videoRecorderProxy.setResolution(2) # Set resolution to VGA (640 x 480)
# We'll save a 5 second video record in /home/nao/recordings/cameras/

tts.say("Recording video")

bas.pauseAwareness()

motion_service.setStiffnesses("Head", 1.0)
names      = "Head"

angleLists = [0*almath.TO_RAD,-7*almath.TO_RAD]

motion_service.angleInterpolationWithSpeed(names, angleLists, 0.6)


videoRecorderProxy.startRecording("/home/nao/recordings/cameras", "gloves")
print "Video record started."

time.sleep(10) # Duration of video 

videoInfo = videoRecorderProxy.stopRecording()
tts.say("Video taken!")
bas.resumeAwareness()
print "Video was saved on the robot: ", videoInfo[1]
print "Total number of frames: ", videoInfo[0]

# os.system("scp nao@10.0.0.83:~/recordings/cameras/gloves.avi ./recordings")
# os.system("scp -P 19563 recordings/test.avi root@2.tcp.ngrok.io:/root/Robots4Hospitals/Gown_doff")
# os.system("scp -P 12432 recordings/gloves.avi root@4.tcp.ngrok.io:/root/Robots4Hospitals/doffing/recordings")

# time.sleep(20)

# os.system("scp -P 19563 root@2.tcp.ngrok.io:/root/Robots4Hospitals/Gown_doff/GownDoffingText .")
# f = open('GownDoffingText', 'r')
# tts.say(f.read())

# os.chdir("recordings")

# os.system("python3 videoToFrames.py")

# vidcap = cv2.VideoCapture('recordings/test.avi')

# sec = 0
# frameCount = videoInfo[0]
# frameRate = 10/videoInfo[0] #//it will capture image in each 0.5 second
# count=1
# success = getFrame(sec)
# while success:
#     count = count + 1
#     sec = sec + frameRate
#     sec = round(sec, 2)
#     success = getFrame(sec)






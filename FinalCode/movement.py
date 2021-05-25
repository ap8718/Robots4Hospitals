import qi
import argparse
import sys
import time
import almath
from naoqi import ALProxy
import numpy as np


def main(session):
    """
    This example uses the angleInterpolation method.
    """
    # Get the service ALMotion.

    tts = ALProxy("ALTextToSpeech", "10.0.0.83", 9559)
    # tts.say("Connected to Aru's computer")

    motion_service  = session.service("ALMotion")

    motion_service.setStiffnesses("Elbow", 1.0)
    motion_service.setStiffnesses("RShoulder", 1.0)
    motion_service.setStiffnesses("LShoulder", 1.0)
    motion_service.setStiffnesses("Wrist", 1.0)
    motion_service.setStiffnesses("ShoulderPitch", 1.0)

    # Example showing a single trajectory for one joint
    # Interpolates the head yaw to 1.0 radian and back to zero in 2.0 seconds
    names      = ["RElbowRoll",
                    "RShoulderPitch",
                    "RShoulderRoll",
                    "LElbowRoll",  
                    "LShoulderPitch",
                    "LShoulderRoll",]
    #              2 angles
    angleLists = [[70.0,70.0],
                    [30.0,30.0],
                    [-30.0,-30.0],
                    [-70.0,-70.0],
                    [30.0,30.0],
                    [30.0,30.0]]
    #              2 times
    timeLists  = [[2.0,8.0],[2.0,8.0],[2.0,8.0],[2.0,8.0],[2.0,8.0],[2.0,8.0]]

    # names = [
    #     "RShoulderPitch",
    #     "RShoulderRoll"
    # ]

    # angleLists = [
    #     [50.], 
    #     [-30.],   #10., 20., 
    # ]

    angleLists  = [[i*(np.pi/180) for i in inner] for inner in angleLists]
    print(angleLists)
   
    #              2 times
    # timeLists  = [[1,2.0,3,4,8.0],[1,2.0,3,4,8.0],[2.0,8.0],[1,2.0,3,4,8.0],[2.0,8.0],[2.0,8.0]]
    # timeLists = [1., 2., 3., 5., 6, 7., 8.]
    # timeLists = [[2.], [2.]]
    isAbsolute = True
    tts.say("Please put your arms up like this")
    print("Putting arms up")
    motion_service.angleInterpolation(names, angleLists, timeLists, isAbsolute)
    print([(180/np.pi)*i for i in motion_service.getAngles("RArm", False)])
  
    time.sleep(1.0)

    # angleLists = [[30., 0.]]
    # angleLists  = [[i*(np.pi/180) for i in inner] for inner in angleLists]
    # timeLists = [1., 2., 3., 4.]
    # print("Putting arms down")
    # motion_service.angleInterpolation(names, angleLists, timeLists, isAbsolute)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="10.0.0.83",
                        help="Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.")
    parser.add_argument("--port", type=int, default=9559,
                        help="Naoqi port number")

    args = parser.parse_args()
    session = qi.Session()
    try:
        session.connect("tcp://" + args.ip + ":" + str(args.port))
    except RuntimeError:
        print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) +".\n"
               "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)
    main(session)
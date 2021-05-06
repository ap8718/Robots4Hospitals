import qi
import argparse
import sys
import time
import almath
from naoqi import ALProxy


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
    angleLists = [[70.0*almath.TO_RAD,70.0*almath.TO_RAD],
                    [30.0*almath.TO_RAD,30.0*almath.TO_RAD],
                    [-30.0*almath.TO_RAD,-30.0*almath.TO_RAD],
                    [-70.0*almath.TO_RAD,-70.0*almath.TO_RAD],
                    [30.0*almath.TO_RAD,30.0*almath.TO_RAD],
                    [30.0*almath.TO_RAD,30.0*almath.TO_RAD]]
    #              2 times
    timeLists  = [[2.0,8.0],[2.0,8.0],[2.0,8.0],[2.0,8.0],[2.0,8.0],[2.0,8.0]]
    isAbsolute = True
    tts.say("Please put your arms up like this")
    motion_service.angleInterpolation(names, angleLists, timeLists, isAbsolute)

    # time.sleep(1.0)
    # names      = ["RElbowRoll",
    #                 "RShoulderPitch",
    #                 "RShoulderRoll",
    #                 "LElbowRoll",  
    #                 "LShoulderPitch",
    #                 "LShoulderRoll",]
    # #              2 angles
    # angleLists = [[0],
    #                 [0.0*almath.TO_RAD],
    #                 [0.0*almath.TO_RAD],
    #                 [0.0*almath.TO_RAD],
    #                 [0.0*almath.TO_RAD],
    #                 [0.0*almath.TO_RAD]]
    # #              2 times
    # timeLists  = [[2.0],
    #                 [2.0],
    #                 [2.0],
    #                 [2.0],
    #                 [2.0],
    #                 [2.0]]
    # isAbsolute = True
    # tts.say("Please put your arms up like this")
    # motion_service.angleInterpolation(names, angleLists, timeLists, isAbsolute)

    time.sleep(1.0)

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
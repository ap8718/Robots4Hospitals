import qi
from naoqi import ALProxy
import sys
import time
import almath

def main(session):

    ms = session.service("ALMotion")
    tts = session.service("ALTextToSpeech")
    # bas = session.service("ALBasicAwareness")
    bas = ALProxy("ALBasicAwareness", "10.0.0.83", 9559)

    # ms.setStiffnesses("Body", 0.0)
    bas.pauseAwareness()

    print("Awareness paused: " + str(bas.isAwarenessPaused()))

    ms.setStiffnesses("LArm", 1.0)
    ms.setStiffnesses("RArm", 1.0)

    names = [
        "RElbowRoll",
        "RElbowYaw",
        "RShoulderRoll",
        "RShoulderPitch",
        "LElbowRoll",
        "LElbowYaw",
        "LShoulderRoll",
        "LShoulderPitch",
    ]

    angles = [
        90.0,
        90.0,
        -90.0,
        0.0,
        -90.0,
        -90.0,
        90.0,
        0.0,
    ]

    angles = [i*almath.TO_RAD for i in angles]
    times = [3.0] * len(names)
    isAbsolute = True

    print("Putting arms up...")
    ms.angleInterpolation(names, angles, times, isAbsolute, _async = True)
    tts.say("Please put your arms up like this")
    time.sleep(1.5)

#   Comment this block out if you don't want to see angles
    angs = ms.getAngles(names, False)
    angs = [i*almath.TO_DEG for i in angs]
    print("Angles: " + str(angs))

    print("Loosening arms...")
    ms.setStiffnesses(names, 0.05)
    time.sleep(2)

    ms.setStiffnesses(names, 0.0)

    # print("Awareness paused: " + str(bas.isAwarenessPaused()))
    bas.resumeAwareness()
    print("Awareness paused: " + str(bas.isAwarenessPaused()))

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
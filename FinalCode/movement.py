import qi
from naoqi import ALProxy
import sys
import time
import almath

def main(session):

    ms = session.service("ALMotion")
    tts = session.service("ALTextToSpeech")

    names = [
        "RElbowRoll",
        "RShoulderRoll",
        "RShoulderPitch",
        "LElbowRoll",
        "LShoulderRoll",
        "LShoulderPitch",
    ]

    ms.setStiffnesses(names, 1.0)

    angles = [
        90.0,
        -90.0,
        0.0,
        -90.0,
        90.0,
        0.0,
    ]

    angles = [i*almath.TO_RAD for i in angles]
    times = [2.0] * len(names)
    isAbsolute = True

    print("Putting arms up...")
    ms.angleInterpolation(names, angles, times, isAbsolute, _async = True)
    tts.say("Please put your arms up like this")
    time.sleep(1.5)

#   Comment this block out if you don't want to see angles
    angs = ms.getAngles(names, False)
    angs = [i*almath.TO_DEG for i in angs]
    print("Angles: " + str(angs))

    print("Putting arms down...")

    angles = [
        0.0,
        0.0,
        90.0,
        0.0,
        0.0,
        90.0,
    ]

    angles = [i*almath.TO_RAD for i in angles]
    times = [2.0] * len(names)

    ms.angleInterpolation(names, angles, times, isAbsolute)

    angs = ms.getAngles(names, False)
    angs = [i*almath.TO_DEG for i in angs]
    print("Angles: " + str(angs))

    ms.setStiffnesses(names, 0.05)

    ms.setStiffnesses(names, 0.0)
    time.sleep(0.5)

    print("All done!")

if __name__ == "__main__":
    session = qi.Session()
    try:
        session.connect("tcp://10.0.0.83:9559")
    except RuntimeError:
        print ("Can't connect to Naoqi at ip '10.0.0.83' on port '9559'.\n"
               "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)
    bas = session.service("ALBasicAwareness")
    bas.pauseAwareness()
    main(session)
    bas.resumeAwareness()

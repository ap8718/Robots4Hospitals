import qi
import os
import time
import donning
import doffing
import sys
import argparse
from datetime import datetime

class HumanGreeter(object):
    """
    A simple class to react to face detection events.
    """

    def __init__(self, app):
        """0
        Initialisation of qi framework and event detection.
        """
        super(HumanGreeter, self).__init__()
        app.start()
        session = app.session
        # Get the service ALMemory.
        self.memory = session.service("ALMemory")
        # Connect the event callback.

        self.subscriber = self.memory.subscriber("WordRecognized")
        self.subscriber.signal.connect(self.on_word_recognised)
        # Get the services ALTextToSpeech and ALFaceDetection.
        self.tts = session.service("ALTextToSpeech")
        self.speech_recognition = session.service("ALSpeechRecognition")
        self.tablet = session.service("ALTabletService")
        self.awareness = session.service("ALBasicAwareness")

        self.speech_recognition.pause(True)

        self.speech_recognition.setVocabulary(["please scan me", "scan doffing please"],False)

        self.speech_recognition.pause(False)
        self.speech_recognition.subscribe("HumanGreeter")
        self.word_recognised = False
        self.session = app.session

    def on_word_recognised(self, value):
        """
        Callback for event FaceDetected.
        """
        if value == []:  # empty value when the face disappears
            self.word_recognised = False
        elif not self.word_recognised:  # only speak the first time a face appears
            self.word_recognised = True
            print value

            if 'please scan me' in value[0] and value[1] > 0.4:
                donning.main(self.session)
            elif 'scan doffing please' in value[0] and value[1] > 0.4:
                doffing.main(self.session)

            self.word_recognised = False

    def run(self):
        """
        Loop on, wait for events until manual interruption.
        """
        print "Starting HumanGreeter"
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print "Interrupted by user, stopping HumanGreeter"
            self.speech_recognition.unsubscribe("HumanGreeter")
            #stop
            sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="10.0.0.83",
                        help="Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.")
    parser.add_argument("--port", type=int, default=9559,
                        help="Naoqi port number")

    args = parser.parse_args()
    try:
        # Initialize qi framework.
        connection_url = "tcp://" + args.ip + ":" + str(args.port)
        app = qi.Application(["HumanGreeter", "--qi-url=" + connection_url])
    except RuntimeError:
        print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) +".\n"
               "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)

    human_greeter = HumanGreeter(app)
    human_greeter.run()
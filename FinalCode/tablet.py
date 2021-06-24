import qi
import time

URL = "https://07c4618a4d76.ngrok.io"

def main(session):
    tabletService = session.service("ALTabletService")

    tabletService.showImageNoCache(URL + "/analysis0.png")

if __name__ == "__main__":
    session = qi.Session()

    try:
        session.connect("tcp://10.0.0.83:9559")
    except:
        print("Could not connect")

    main(session)

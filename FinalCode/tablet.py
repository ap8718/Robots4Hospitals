import qi 
import time

def main(session):
    tabletService = session.service("ALTabletService")
    
    tabletService.showImageNoCache("http://f5f56144bb07.ngrok.io/analysis0.png")


    # time.sleep(10)

    # tabletService.hideImage()

if __name__ == "__main__":
    session = qi.Session()

    try:
        session.connect("tcp://10.0.0.83:9559")
    except:
        print("Could not connect")
    
    main(session)
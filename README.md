# Robots4Hospitals

## Year 3 MEng Electrical Engineering Group Project 2020-2021

Welcome to the Robots4Hospitals repository! For our Year 3 MEng Electrical and Electronics Engineering Group Project at Imperial College London, we have developed an autonomous robot-based system to perform full inspection of both the PPE donning and doffing processes.

- [Leaflet](https://imperiallondon.sharepoint.com/sites/Robots4Hospitals-EE/Shared%20Documents/General/Project%20Leaflet.pdf)
- [Video](https://imperiallondon.sharepoint.com/sites/Robots4Hospitals-EE/Shared%20Documents/General/R4HVideo.mp4)
- [Presentation](https://imperiallondon.sharepoint.com/:p:/s/2021ThirdYearM.EngGroupProjects-EE/Ef8rAftzq0tDg2MAxeCH-n4BwFn833gjoIQOeGts3Vd6Bw?e=VMcXhn)
- [Documentation]()

## Requirements

You MUST have a Linux machine to connect to Pepper the robot.
- Linux machine
- [Python >= 3.6.0](https://www.python.org/downloads/) (for Computer Vision models)
- [Python 2.7](https://www.python.org/downloads/) on Linux machine (for Robot)
- [Pepper Python SDK](http://doc.aldebaran.com/2-5/dev/python/install_guide.html)
- [OpenCV](https://pypi.org/project/opencv-python/)
- [Tensorflow Object Detection API](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html)
- [MediaPipe](https://google.github.io/mediapipe/getting_started/python.html)
- [ngrok](https://ngrok.com/download) (You must also make an account. Free account is sufficient)
- [YOLO5](https://github.com/ultralytics/yolov5)

```shell
$ git clone https://github.com/ap8718/Robots4Hospitals
$ cd Robots4Hospitals
$ pip3 install -r requirements_cpu.txt (if using cpu)
$ pip3 install -r requirements_gpu.txt (if using gpu)
```


## Connecting to Pepper
First, you must install the Python SDK on whichever terminal you are using to connect to Pepper. Change directory to Robots4Hospitals and run the following command to export the Python path to the within the python-sdk folder:

```shell
$ export PYTHONPATH=${PYTHONPATH}:python-sdk/lib/python2.7/site-packages
```
Verify that the installation is successful by importing the `qi` or `naoqi` library:

```shell
$ python2 -c "import qi"
```
If no error message comes up, the installation has been successful. Note that this only works on Linux. Please check the [Installation Guide](http://doc.aldebaran.com/2-5/dev/python/install_guide.html) for installation on a different OS, although there is no guarantee of success.

Next, boot up Pepper by pressing the power button located on its chest behind the tablet. The booting sequence should take about a minute. Once Pepper is booted, please press (do not hold) the power button. It should say "Hello, I'm Pepper. My internet address is <ROBOT_IP>". This is the IP you will use to connect to Pepper. The port will always be 9559 with Pepper. If Pepper says "I can't connect to the network" please reboot or hard reboot the robot until connection is established.

The program should be run from a terminal in the same directory as `main.py`. Change directory into `FinalCode/` and run `main.py` as shown in the following.

```shell
$ python2 main.py --ip '<ROBOT_IP>'
```

## Connecting to the GPU
Once ngrok is installed and your authtoken is available, run the following code to add your authtoken to the configuration file:

```shell
$ ./ngrok authtoken <YOUR_AUTHTOKEN>
```
Then run this [Colab notebook](https://colab.research.google.com/drive/1thf0PNDGo3MBUKt8xfW7kUeiBnhA2Yke#scrollTo=PlemhodHuWYv) to get the IP and port of the GPU. Then SSH into the GPU machine with the following:
```shell
$ ssh root@<GPU_IP> -p <GPU_PORT>
```
then enter `password` as the password. This can be modified in the Colab script above.

**Make sure to change the IP and port of the remote machine in `doffing.py` and `takePhotoGPU.py` for the file transfer to work.**

Then, on a local machine, set up passwordless access to this remote machine. If you do not already have an SSH key, (run `cat ~/.ssh/id_rsa.pub` to check) run the following command and press enter when prompted:
```shell
$ ssh-keygen
```
Then run the following to allow passwordless access to the remote GPU machine. This will be important when autonomously transferring files between local and remote machines:
```shell
$ ssh-copy-id root@<GPU_IP> -p <GPU_PORT>
```
It is essential that this is done particularly for the Linux machine that is connected to the robot.
Once SSHed into the remote GPU machine, from a local machine and from the same directory as `gpuInit.py`, securely copy the file to the remote machine with the following:
```shell
$ scp -P <GPU_PORT> gpuInit.py root@<GPU_IP>:
```
If passwordless access has been successfully configured, the file transfer should happen without a prompt for a password.
Then, in your remote machine, run `gpuInit.py` to install all the necessary files and libraries:
```shell
$ python gpuInit.py
```
Now, in your remote machine, change directories to `completeGPULoop.py` and run it on python while running `main.py` on the Linux machine:
```shell
$ python gpuLoop.py
```
You should now be able to say the key phrases and start the analysis for donning or doffing.

## Displaying images on the tablet
To display the image taken by Pepper on its tablet, we must publicly host the image. We will use ngrok to publicly host the image.

First, on a separate terminal window, change directory to `imagesFromPepper/` and start a localhost server on port 80:
```shell
$ python3 -m http.server 80
```
for python 2:
```shell
$ python2 -m SimpleHTTPServer 80
```
Verify that the server is running by searching for `http://localhost/` on a web browser.
Next, on another terminal window, run the HTTP tunneling to publicly host the contents of the local server:
```shell
$ ./ngrok http 80
```
Verify that the server is running by pasting the URL displayed next to "Forwarding" (boxed in red) and seeing if it shows the same content.

![Screenshot of ngrok HTTP interface with URL highlighted in red](https://github.com/ap8718/Robots4Hospitals/blob/main/FinalCode/imagesFromPepper/ngrokHTTPurl.png)

Lastly, in `tablet.py`, change the URL to the one currently being used to publicly host your localhost server.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

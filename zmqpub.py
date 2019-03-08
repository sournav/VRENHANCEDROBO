import zmq
import cv2
import time
#port = "5556"
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.connect("tcp://127.0.0.1:5556")
boi = b'\0xff\0xb0'
while True:
    print(boi)
    socket.send(boi)
    time.sleep(1)
    

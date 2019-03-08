import zmq
import cv2
import time
port = "5556"
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.bind("tcp://127.0.0.1:5556")
socket.setsockopt_string(zmq.SUBSCRIBE, '')
while True:
    print(socket.recv())
    

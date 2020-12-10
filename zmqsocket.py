import socket
import zmq


import sys
import threading
import time
import struct

def send_msg(sock, msg):
    # Prefix each message with a 4-byte length (network byte order)
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

def recv_msg(sock):
    # Read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # Read the message data
    return recvall(sock, msglen)

def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


class TcpPublisher :
    def __init__(self,_HOST=None,_PORT=50007,topic_name='Unknown') :
        self.HOST = _HOST               # Symbolic name meaning all available interfaces
        self.PORT = _PORT              # Arbitrary non-privileged port
        self.topic_name =topic_name
        socketType = zmq.PUB
        self.context = zmq.Context()
        self.socket = self.context.socket(socketType)
        self.socket.bind('tcp://{}:{}'.format(self.HOST,self.PORT))


    def send(self,data) :
        # print(bytes('{}~{}'.format(self.topic_name,data).encode()))
        self.socket.send(bytes('{}~{}'.format(self.topic_name,data).encode()))
        # print('Send message Complete')

        

class TcpSubscriber :
    def __init__(self,_HOST=None,_PORT=50007,topic_name='Unknown') : 
        self.topic_name = topic_name
        self.HOST = _HOST               # Symbolic name meaning all available interfaces
        self.PORT = _PORT
        socketType = zmq.SUB
        self.context = zmq.Context()
        self.socket = self.context.socket(socketType)
        self.socket.connect('tcp://{}:{}'.format(self.HOST,self.PORT))

    def connect(self,address) :
        self.socket.setsockopt(zmq.SUBSCRIBE,bytes(self.topic_name.encode()))
        self.socket.connect(address)
        self.socket.subscribe(b'')

    def receive(self) :
        return self.socket.recv().decode().split('~')
        

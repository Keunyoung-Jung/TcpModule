import socket
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


class TcpServer :
    def __init__(self,_HOST=None,_PORT=50007) :
        self.HOST = _HOST               # Symbolic name meaning all available interfaces
        self.PORT = _PORT              # Arbitrary non-privileged port
        self.s = None
        self._stop = False
        self._data_ready = threading.Event()
        self._thread = threading.Thread(target=self._run,args=())
        self._thread.daemon = True
        self._thread.start()
        self._tmp = None
        self._data = None

    def receive(self,timeout=15.0) :
        # flag = self._data_ready.wait()
        # if not flag:
        #     raise TimeoutError(
        #         "Timeout while reading from subscriber")
        # self._data_ready.clear()
        if self._tmp != self._data :
            print('Server Recived Message : {}'.format(self._data))
            self._tmp = self._data
            return self._data
        else :
            pass

    def send(self,data) :
        send_msg(self.conn,data)
        # print('Send message Complete')

    def _run(self) :
        for res in socket.getaddrinfo(self.HOST, self.PORT, socket.AF_UNSPEC,
                                    socket.SOCK_STREAM, 0, socket.AI_PASSIVE):
            af, socktype, proto, canonname, sa = res
            try:
                self.s = socket.socket(af, socktype, proto)
            except OSError as msg:
                self.s = None
                continue
            try:
                self.s.bind(sa)
                self.s.listen(1)
            except OSError as msg:
                self.s.close()
                self.s = None
                continue
            break
        if self.s is None:
            print('could not open socket')
            sys.exit(1)
        while not self._stop :
            self.conn, addr = self.s.accept()
            with self.conn:
                print('TCP socket Connected by', addr)
                time.sleep(1)
                while not self._stop:
                    try :
                        self._data = recv_msg(self.conn)
                    except :
                        print('Time out Connect, Reconnect now')
                    # if not self._data: break
                    self._data_ready.set()
            self.close()
    def close(self) :
        print('Closed Server')
        self._stop = True
        

class TcpClient :
    def __init__(self,_PORT=50007) :
        self.HOST = socket.gethostbyname(socket.gethostname())   # The remote host
        self.PORT = _PORT              # The same port as used by the server
        self.s = None
        self._stop = False
        self._data_ready = threading.Event()
        self._thread = threading.Thread(target=self._run,args=())
        self._thread.daemon = True
        self._thread.start()
        self._tmp = None
        self._data = None

    def receive(self) :
        # flag = self._data_ready.wait()
        # if not flag:
        #     raise TimeoutError(
        #         "Timeout while reading from subscriber")
        # self._data_ready.clear()
        if self._tmp != self._data :
            print('Client Received', repr(self._data))
            self._tmp = self._data
            return self._data
        else :
            pass

    def send(self,data) :
        # print('Send message Complete')
        send_msg(self.s,data)

    def close(self) :
        print('Closed Client')
        self._stop = True

    def _run(self) :
        for res in socket.getaddrinfo(self.HOST, self.PORT, socket.AF_UNSPEC, socket.SOCK_STREAM):
            af, socktype, proto, canonname, sa = res
            try:
                self.s = socket.socket(af, socktype, proto)
            except OSError as msg:
                self.s = None
                continue
            try:
                self.s.connect(sa)
            except OSError as msg:
                self.s.close()
                self.s = None
                continue
            break
        if self.s is None:
            print('could not open socket')
            sys.exit(1)
        with self.s:
            while not self._stop :
                try:
                    self._data = recv_msg(self.s)
                except :
                    print('Time out Connect, Reconnect now')
                self._data_ready.set()
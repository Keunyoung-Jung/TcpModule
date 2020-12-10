import sys
import time
import traceback
import numpy as np
import cv2
from imutils.video import FPS
import imagezmq
import socket
import threading

class VideoStreamSubscriber:

    def __init__(self, hostname, port):
        self.hostname = hostname
        self.port = port
        self._stop = False
        self._data_ready = threading.Event()
        self._thread = threading.Thread(target=self._run, args=())
        self._thread.daemon = True
        self._thread.start()

    def receive(self, timeout=15.0):
        flag = self._data_ready.wait(timeout=timeout)
        if not flag:
            raise TimeoutError(
                "Timeout while reading from subscriber tcp://{}:{}".format(self.hostname, self.port))
        self._data_ready.clear()
        return self._data

    def _run(self):
        receiver = imagezmq.ImageHub("tcp://{}:{}".format(self.hostname, self.port), REQ_REP=False)
        receiver.connect('tcp://192.168.0.142:5555')
        receiver.connect('tcp://192.168.0.140:5555')
        receiver.connect('tcp://172.17.0.3:5555')
        receiver.connect('tcp://192.168.0.250:5555')
        while not self._stop:
            self._data = receiver.recv_jpg()
            self._data_ready.set()
        receiver.close()

    def close(self):
        self._stop = True

count = 0
hostip = socket.gethostbyname(socket.gethostname())
port = '5555'

# image_hub = imagezmq.ImageHub(open_port='tcp://{}:{}'.format(hostip,port),REQ_REP=True)
# print('Try Connect broadcast Camera ..')
# # image_hub.connect('tcp://192.168.0.142:5555')
# # image_hub.connect('tcp://192.168.0.140:5555')
# image_hub.connect('tcp://172.17.0.4:5555')
# print('connect')
receiver = VideoStreamSubscriber(hostip, port)


connected_cam1 = False
connected_cam2 = False

while True:  # receive images until Ctrl-C is pressed
    st = time.time()
    # sent_from, jpg_buffer = image_hub.recv_jpg()
    sent_from,jpg_buffer = receiver.receive()

    image = cv2.imdecode(np.frombuffer(jpg_buffer, dtype='uint8'), -1)
    # print(time.time()-st)
    if sent_from == 'rpi1' :
        if not connected_cam1 :
            print('Connected Camera-1 !!')
            connected_cam1 = True
        image_govis = image.copy()
    else :
        if not connected_cam2 :
            print('Connected Camera-2 !!')
            connected_cam2 = True
        image_gopiz = image.copy()
    # see opencv docs for info on -1 parameter
    cv2.imshow(sent_from+'2', image)  # display images 1 window per sent_from
    k = cv2.waitKey(1)
    # image_hub.send_reply(b'OK')
    if k == ord(' ') :
        cv2.imwrite('/media/govis/extra/data_voucher/future_raspi/gopiz_{}.jpg'.format(count),image_gopiz)
        cv2.imwrite('/media/govis/extra/data_voucher/future_raspi/govis_{}.jpg'.format(count),image_govis)
        print('Save complete {}th image'.format(count))
        count += 1
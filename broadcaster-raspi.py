import sys
import socket
import time
import traceback
import cv2
import imagezmq
import threading

# use either of the formats below to specifiy address of display computer
# sender = imagezmq.ImageSender(connect_to='tcp://192.168.0.142:5555')
# sender = imagezmq.ImageSender(connect_to='tcp://192.168.1.190:5555')
print('Broadcast Loading ..')
rpi_name = 'rpi1'  # send RPi hostname with each image
cap = cv2.VideoCapture(0)
cap.set(3,1920)
cap.set(4,1080)
cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
# time.sleep(2.0)  # allow camera sensor to warm up
print('Start Broadcast !!')
jpeg_quality = 95  # 0 to 100, higher is better quality, 95 is cv2 default


class Send_image:
    def __init__(self) :
        self._stop = False
        self._sender = None
        self.rpi_name = 'rpi'
        self._ready = False
        self._sendsw = True
        self._thread = threading.Thread(target=self._run,args=())
        self._thread.daemon=True
        self._thread.start()
        self.count = 0

    def send_image(self,image) :
        st = time.time()
        ret_code, jpg_buffer = cv2.imencode(
           ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        sender.send_jpg(self.rpi_name, jpg_buffer)
        self._sendsw = True
        #print('send fps {}'.format(1/(time.time()-st)))

    def clear_image_buffer(self,image) :
        #print('clear')
        if self._sendsw :
            self._image = image
            self._ready = True
            self.count = 0

    def _run(self) :
        self.sendsw = True
        while not self._stop :
            if self._ready :
                self.sendsw = False
                self.send_image(self._image)
                self._ready = False

    def close(self):
        self._stop = True

try:
    with imagezmq.ImageSender(connect_to='tcp://*:5555',REQ_REP=False) as sender:
        image_sender = Send_image()
        while True:  # send images as stream until Ctrl-C
            st = time.time()
            ret,image = cap.read()
            #print('read fps {}'.format(1/(time.time()-st)))
            image_sender.clear_image_buffer(image)
            #print('clear fps {}'.format(1/(time.time()-st)))
            # above line shows how to capture REP reply text from Mac
except (KeyboardInterrupt, SystemExit):
    pass  # Ctrl-C was pressed to end program
except Exception as ex:
    print('Python error with no Exception handler:')
    print('Traceback error:', ex)
    traceback.print_exc()
finally:
    sys.exit()

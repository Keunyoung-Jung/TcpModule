#-*-coding: utf-8-*-
import time
import numpy as np
#import argparse
import cv2
import rospy
from std_msgs.msg import String
import sys

from sort import *
from pizza.pizza_maker import *
import darknetTR

import logging
import traceback
logging.basicConfig(level=logging.ERROR)
import socket
import threading
import imagezmq
import json
import urllib
import base64
from zmqsocket import TcpSubscriber,TcpPublisher

rpi1_ip = '192.168.0.248'

class VideoStreamSubscriber:

    def __init__(self, hostname, port):
        self.hostname = hostname
        self.port = port
        self._stop = False
        self._data_ready = threading.Event()
        self._thread = threading.Thread(target=self._run, args=())
        self._thread.daemon = True
        self._thread.start()
        self.startsw = True

    def receive(self, timeout=15.0):
        if self.startsw :
            flag = self._data_ready.wait(timeout=timeout)
            if not flag:
                raise TimeoutError(
                    "Camera is not open / Time out tcp://{}:{}".format(self.hostname, self.port))
            self.startsw = False
        # print('receive!!')
        # self._data_ready.clear()
        return self._data

    def _run(self):
        receiver = imagezmq.ImageHub("tcp://{}:{}".format(self.hostname, self.port), REQ_REP=False)
        receiver.connect('tcp://{}:5555'.format(rpi1_ip))
        while not self._stop:
            self._data = receiver.recv_jpg()
            # print(self._data)
            self._data_ready.set()
            # print('images!!')
        receiver.close()

    def close(self):
        self._stop = True

#bounding box info
class Detect:
	def __init__(self, dets, asdets):
		self.dets = dets
		self.asdets = asdets

#tracker info
class SORT :
	def __init__(self, tracker, memory) :
		self.tracker = tracker
		self.memory = memory

#topping count info
class tp_table :
	def __init__(self,tp_count) :
		self.tp_count = tp_count
		self.ROI = None
		self.score = 100
		self.topping_size = []
		self.finish_time = 0
		self.start_time = 0
		self.topping_boxes = None

class View :
	def __init__(self) :
		self.save_view = None
		self.pre_center = None
		self.pre_box = None

class Message :
	def __init__(self) :
		self.image = None
		self.status = None
		self.name = None
		self.pct = None
		self.count = None
		self.currValue = None
		self.nextValue = None
		self.pizza_recipe = None
		self.cooking_time = None
		self.finish_time = None
		self.curr_score = None
		self.finish_score = None

class Reset :
	def __init__(self) :
		self.reset_threshold = 120
		self.timer = time.time()
		self.time_checker = 0
		self.dough_checker = False
		self.finish_counter = 15


main_path = './'
#test_path = main_path+'video_test/split_video/'
#test_path = main_path+'pizza_video/'

#file = 'sample2_split_1.mp4'	#select test video
#file = 'cheese_pizza.mp4'

write_toggle = False #if you want recode video

class_num = 21	#class number

count = 0	#frame count

threshold = 0.5	#detect threshold

wait_time = 0

wait_sw = True

pizza = None

sw = True	#video fast forward switch (press key']')

# rospy.init_node('dough_detector', anonymous=True)
# ros = rospy.Publisher('/gopiz_dough_status',String,queue_size= 1)

font = cv2.FONT_HERSHEY_DUPLEX

term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 ) #meanshift termination(not used)

np.random.seed(42)	#color random seed
# net = darknet.load_net(bytes(main_path+'cfg/ky_yolov3.cfg',encoding='utf-8'), 
# 					   bytes(main_path+'weight/comla_default_yolov3_172000.weights',encoding='utf-8'),0)

# meta = darknet.load_meta(bytes(main_path+'data/ky_obj.data',encoding='utf-8'))

cheese_model = cheese_reg.load_regmodel()

moose_model = moose_reg.load_regmodel()

cheddar_model = cheddar_reg.load_regmodel()

teriyaki_model = teriyaki_reg.load_regmodel()

detector = darknetTR.YOLO4RT(weight_file=main_path+'detector_models/20201003_yolov4x704_fp32.rt',
						metaPath=main_path+'detector_models/ground.data',nms=0.2,conf_thres=0.3)

now = ''

labels_arr = ['Dough',
				'onion',
				'mushroom',
				'blackOlive',
				'pepperoni',
				'corn',
				'bacon',
				'bulgogi',
				'wedgePotato',
				'halfSlicedHam',
				'bellPepper',
				'tomato',
				'pineapple',
				'garlicChip',
				'fireChicken',
				'sweetPotatoCube',
				'steak',
				'teriyakiOnion',
				'shrimp',
				'crab',
				'gorgonzola']

sort = []
detected = []
COLORS = []
tt = tp_table(np.zeros((class_num,),dtype = int))
reset = Reset()
view_obj = View()
msg = Message()
pizza = None

#create object
for i in range(class_num) :
	sort.append(SORT(Sort(max_age=600,min_hits=1),{}))
	detected.append(Detect([],np.asarray([])))
	COLORS.append((np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255)))

# fourcc = cv2.VideoWriter_fourcc(*'XVID')

# test_path = main_path+'20200925_test_vid/'

# file = 'tcenter_gorgonzola.mp4'

# vs = cv2.VideoCapture(test_path+file)
hostip = socket.gethostbyname(socket.gethostname())
port = '5555'
receiver = VideoStreamSubscriber(hostip,port)
# cap = cv2.VideoCapture('/home/govis/master/toppingtable-multi/gourmet_table_models/20200925_test_vid/tcenter_combination.mp4')
#bounding box info

def pizza_selector(pizza_name) :
	global pizza

	if pizza_name is not None :
		pizza = Pizza(pizza_name)
		pizza.tp_count = tt.tp_count
		pizza.moose_model = moose_model
		pizza.cheddar_model = cheddar_model
		pizza.teriyaki_model = teriyaki_model

#After yolo detections, return Bounding box
def retbox(detections,i,labels_arr) :
	global pizza
	#print(detections)
	score = detections[i][1]
	classes = detections[i][0]
	label =labels_arr[classes]

	x1 = int(round((detections[i][2][0]))) # top left x1 
	y1 = int(round((detections[i][2][1]))) # top left xy 
	x2 = int(round((detections[i][2][0]) + (detections[i][2][2]))) # bottom right x2 
	y2 = int(round((detections[i][2][1]) + (detections[i][2][3]))) # bottom right y2 

	# print(pizza.name,'////',label)
	if pizza is not None :
		if pizza.name == 'Sweet_potato' and label == 'wedgePotato' :
			label = 'sweetPotatoCube'
			classes = 15
     
	box = np.array([x1,y1,x2,y2])

	return label, score, box ,classes

#draw bounding box
def draw(dets,dets_yolo , frame,view, tracker ,memory ,label ,idx ,color ,font, tt,cheese_model) :

	if len(dets) != 0:
		tracks = tracker.update(dets)

		#print(dets)
		#print(tracks)
		det_boxes = []
		indexIDs = []
		view_calc = []
		memory = {}

		for track in tracks:
			det_boxes.append([track[0], track[1], track[2], track[3]])
			indexIDs.append(int(track[4]))
			view_calc.append(abs(frame.shape[1]/2-((track[0]+(track[2]-track[0])/2))))
			memory[indexIDs[-1]] = det_boxes[-1]

		if len(det_boxes) > 0:
			for box in det_boxes:
				(x, y) = (int(box[0]), int(box[1]))
				(w, h) = (int(box[2]), int(box[3]))

				#split dough in action
				if label == 'Dough' :
					tt.ROI = x,y,w,h
					reset.timer = time.time()
					reset.dough_checker = True
					cv2.rectangle(frame, (x, y), (w, h), color[idx], 2)

					#Nearest point to center of frame
					if det_boxes.index(box) == view_calc.index(min(view_calc)) :
						msg.name, msg.pct , msg.currValue , msg.nextValue,msg.count = pizza.maker(frame,view,font,color,idx,tt,cheese_model)
						#It should also be located in the center of the frame
						view_obj.save_view = (x,y,w,h)
					break
				else :
					reset.time_checker = time.time() - reset.timer

	if len(dets_yolo) != 0:
		for box in dets_yolo :
			(x, y) = (int(box[0]), int(box[1]))
			(w, h) = (int(box[2]), int(box[3]))
			centerx, centery = x+((w-x)/2) , y+((h-y)/2)
				

			if tt.ROI is not None :
				if tt.ROI[0] < centerx < tt.ROI[2] and tt.ROI[1] < centery < tt.ROI[3] :
					tt.tp_count[idx] += 1 


	else : 
		reset.time_checker = time.time() - reset.timer
		if msg.currValue is None :
			msg.pct , msg.currValue , msg.nextValue = 0 , None , None


def show_view(view,view_obj,font,color,pizzaname) :
	try :
		x,y,w,h = view_obj.save_view
		dough_center_x, dough_center_y = x+((w-x)/2) , y+((h-y)/2)
		
		center_x_min = dough_center_x - 100
		center_x_max = dough_center_x + 100
		center_y_min = dough_center_y - 100
		center_y_max = dough_center_y + 100

		if view_obj.pre_center == None :
			view_obj.pre_center = (dough_center_x,dough_center_y)
			view_obj.pre_box = (x,y,w,h)
		else :
			if center_x_min > view_obj.pre_center[0] or center_x_max < view_obj.pre_center[0] :
				view_obj.pre_center = (dough_center_x,dough_center_y)
				view_obj.pre_box = (x,y,w,h)
			elif center_y_min > view_obj.pre_center[1] or center_y_max < view_obj.pre_center[1] :
				view_obj.pre_center = (dough_center_x,dough_center_y)
				view_obj.pre_box = (x,y,w,h)


		view_x1 = view_obj.pre_box[0] - 50
		view_x2 = view_obj.pre_box[2] + 50
		view_y1 = view_obj.pre_box[1] - 50
		view_y2 = view_obj.pre_box[3] + 50

		if pizza.sauce_toggle == True :
			cv2.putText(view,pizzaname,(10,70),font,2,color[0],2)
		if pizza.cheese_toggle == True :
			cv2.putText(view,'Spread Mozzarella Cheese',(10,70),font,2,color[1],2)

		return view[view_y1:view_y2,view_x1:view_x2]
	except :
		pass

def capture(pizza_name) :
	global main_path,test_path,write_toggle,class_num,count
	global threshold,sw,font,term_crit,net,meta,cheese_model,moose_model
	global labels_arr,sort,detected,COLORS,tt,view_obj
	global file,pizza,fourcc,vs,prop,total,msg , wait_time, wait_sw
	global now
	
	if pizza is None :
		pizza_selector(pizza_name)

	else : 
		if tt.start_time == 0 :
			tt.start_time = time.time()

	try :
		# st = time.time()
		sent_from, jpg_buffer = receiver.receive()
		# print(sent_from)
		frame = cv2.imdecode(np.frombuffer(jpg_buffer, dtype='uint8'), -1)
		# print('fps {} '.format(1/(time.time()-st)))
		cv2.imshow('dd',frame)

		view = frame.copy()
		view2 = frame.copy()
		count += 1
		
		sw = False
		start = time.time()
		#refresh detections
		for i in range(class_num) :
			detected[i].dets = []
			detected[i].asdets = np.asarray([])

		tt.tp_count = np.zeros((class_num,),dtype = int)
		rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		detections = detector.detect(rgb_frame)

		#Receive detections info
		for i in range(len(detections)) :
			_ , score , box ,classes = retbox(detections,i,labels_arr)
			if score < threshold :
				break
			else :
				left,top,right,bottom=box
				detected[classes].dets.append([left,top,right,bottom])
				if classes == 0 :
					detected[classes].asdets = np.asarray(detected[classes].dets)

		#Tracking and Draw Bounding Box

		reset.dough_checker = False
		tt.topping_boxes = detected

		if pizza is not None :
			for i in range(class_num) :
				draw(detected[i].asdets,detected[i].dets, frame,view, sort[i].tracker, sort[i].memory ,
					labels_arr[i],i, COLORS ,font, tt,cheese_model)
			view = show_view(view,view_obj,font,COLORS,pizza.name)
			sw = True
			pizza.tp_count = tt.tp_count
			msg.pizza_recipe = pizza.recipe
			tt.topping_size = []
			if reset.dough_checker == False :

				if pizza.finish_toggle == True :
					reset.finish_counter -= 1

			if pizza.finish_toggle == True :
				msg.finish_time = tt.finish_time
				msg.finish_score = tt.score
			
			if reset.time_checker > reset.reset_threshold :
				pizza = None
				reset.time_checker = 0
				reset.timer = time.time()
				tt.score = 100
				tt.start_time = 0
				tt.finish_time = 0
				msg.finish_time = None
				msg.finish_score = None
				msg.name, msg.pct , msg.currValue , msg.nextValue = None,None,None,None

			elif pizza.finish_toggle == True and reset.dough_checker == False and reset.finish_counter <= 0:

				pizza = None
				reset.time_checker = 0
				reset.timer = time.time()
				reset.finish_counter = 15
				tt.score = 100
				tt.start_time = 0
				tt.finish_time = 0
				msg.finish_time = None
				msg.finish_score = None
				msg.name, msg.pct , msg.currValue , msg.nextValue = None,None,None,None
				
			cooking_minute = int((time.time()-tt.start_time)//60)
			cooking_second = int((time.time()-tt.start_time)%60)
			if cooking_minute > 100  or cooking_second > 100 :
				cooking_minute = 00
				cooking_second = 00

		else :
			cooking_minute = 00
			cooking_second = 00
		cv2.putText(frame,'cooking time : {}:{}'.format(cooking_minute,cooking_second),(frame.shape[1]-420,120),font,1,(0,255,0),2)
		msg.cooking_time = '{}:{}'.format(cooking_minute,cooking_second)
		msg.curr_score = tt.score

		table_bg = np.zeros((1100,500,3), np.uint8) + 255
		for i in range(class_num) :
			cv2.putText(table_bg,'{} : {} EA'.format(labels_arr[i],tt.tp_count[i]),(10,int(frame.shape[0]*i/class_num+1)+30),font,1,(0,0,0),2)


		encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
		try :
			if reset.reset_threshold-reset.time_checker < 11 :
				cv2.putText(view,'Reset',(int(view.shape[1]/2)-80,int(view.shape[0]/2)-80),font,5,(0,255,0),2)
				cv2.putText(view,'{}'.format(int(reset.reset_threshold-reset.time_checker)),(int(view.shape[1]/2)-10,int(view.shape[0]/2)-5),font,7,(0,255,0),5)
			
			res_img = cv2.resize(view,(480,270))
			_, encimg = cv2.imencode('.jpeg', res_img, encode_param)
		except :
			if reset.reset_threshold-reset.time_checker < 11 :
				cv2.putText(view2,'Reset',(int(view2.shape[1]/2)-80,int(view2.shape[0]/2)-80),font,5,(0,255,0),2)
				cv2.putText(view2,'{}'.format(int(reset.reset_threshold-reset.time_checker)),(int(view2.shape[1]/2)-10,int(view2.shape[0]/2)-5),font,7,(0,255,0),5)
			res_img2 = cv2.resize(view2,(480,270))
			_, encimg = cv2.imencode('.jpeg', res_img2, encode_param)
			print('Not detected Dough yet')
		msg.image = encimg
		
		# if msg.currValue is not None :
		# 	ros.publish(msg.currValue.split(':')[0])
		return msg ,sw
	except Exception as ex:
		print(logging.error(traceback.format_exc()))
		print('[Error] {}'.format(ex))

def message_converter(message,send_arr) :

	order_cpu_time,order_time,order_list,user_name,user_store = send_arr

	if message.currValue is not None :
		cur = message.currValue.split(':')

		cur_name,cur_count,cur_unit = cur[0],cur[1],cur[2] 

	else :
		cur_name,cur_count,cur_unit = None,None,None

	if message.nextValue is not None :
		nex = message.nextValue.split(':')

		next_name,next_count,next_unit = nex[0],nex[1],nex[2]

	else :
		next_name,next_count,next_unit = None,None,None
	
	content = 'data:image/jpeg;base64,' + urllib.parse.quote(base64.b64encode(message.image))
	for data in order_cpu_time :
		order_minute = int((time.time()-data)//60)
		order_second = int((time.time()-data)%60)
		order_time.append('{}:{}'.format(order_minute,order_second))

	msg = {
            "cam_image" : content,
            "pizza_name" : message.name,
            "pizza_recipe" : message.pizza_recipe,
            "order_list" : order_list,
            "order_time" : order_time,
            "curr_topping" : {
                            "name" : cur_name,
                            "status" : message.pct,
                            "goal" : cur_count,
                            "measure" : cur_unit
                            },
            "next_topping" : {
                            "name" : next_name,
                            "goal" : next_count,
                            "measure" : next_unit,
                            },
            "cooking_time" : message.cooking_time,
            "finish_time" : message.finish_time,
            "curr_score" : message.curr_score,
            "finish_score" : message.finish_score,
            "user_name" : user_name,
            "user_store" : user_store
        }
	return json.dumps(msg,default=str)

class Order_viewer :
	def __init__(self,pub_adr,pub_port) :
		self.receiver = TcpSubscriber(pub_adr,pub_port,topic_name='order_info')
		# self.receiver.connect('tcp://{}:{}'.format(pub_adr,pub_port))
		self._thread = threading.Thread(target=self._run,args=())
		self._thread.daemon = True
		self._thread.start()
		self._data = None

	def receive(self):
		return self._data
	
	def _run(self) :
		while True :
			self._data = self.receiver.receive()
		receiver.close()

def start() :
	order_data = None
	pizza_name = None
	#order_cpu_time,order_time,order_list,user_name,user_store
	order_cpu_time = []
	order_time = []
	order_list = []
	user_name = None
	user_store = None
	ws_receiver = Order_viewer(rpi1_ip,50008)
	ws_sender = TcpPublisher(hostip,50009,topic_name='yolo_info')
	print('Receiver Opened tcp://{}:{}'.format(rpi1_ip,50008))
	print('Sender Opend tcp://{}:{}'.format(hostip,50009))
	while True :
		st = time.time()
		socket_msg = ws_receiver.receive()
		# print(socket_msg)
		socket_msg = None
		if socket_msg is not None and not b'' :
			topic,socket_message = socket_msg
			if topic == 'order_info' :
				order_data = json.loads(socket_message)
				pizza_name = order_data['order_menu_name']
				#order_cpu_time,order_time,order_list,user_name,user_store
				order_cpu_time = []
				order_time = []
				order_list = []
				user_name = order_data['assigned_worker']
				user_store = 'test'

		message, switch = capture(pizza_name)
		send_arr = (order_cpu_time,order_time,order_list,user_name,user_store)
		json_msg = message_converter(message,send_arr)
		# print('fps {} '.format(1/(time.time()-st)))
		# print(sys.getsizeof(json_msg))

		ws_sender.send(json_msg)


if __name__ == "__main__" :
	
	start()
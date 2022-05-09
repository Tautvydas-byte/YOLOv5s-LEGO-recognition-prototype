import torch  # upload YOLO model and make detections
from matplotlib import pyplot as plt  # for visualising images
import numpy as np  # for array transformation
import cv2  # helps access the webcam and render feeds
import seaborn as sn

#model load
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp54/weights/best.pt',
					   force_reload=True)  # https://pytorch.org/hub/ultralytics_yolov5/

# access for webcam in real time.
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(width, height)

while cap.isOpened():
	ret, frame = cap.read()

	# make Detections
	results = model(frame)

	cv2.imshow('YOLOv5s_Lego_recognition', np.squeeze(results.render()))  # name of the render frame

	if cv2.waitKey(10) & 0xFF == ord('q'):
		break
cap.release()  # release webcam
cv2.destroyAllWindows()

# Using mobile camera
# cap = cv2.VideoCapture("https://192.168.1.205:8080/video")
# while cap.isOpened():
# 	ret, frame = cap.read()
# 	resized = cv2.resize(frame, (320, 240))
# 	# make Detections
# 	results = model(resized)
#
# 	cv2.imshow('YOLOv5s_Lego_recognition', np.squeeze(results.render()))  # name of the render frame
#
# 	if cv2.waitKey(10) & 0xFF == ord('q'):
# 		break
# cap.release()  # release webcam
# cv2.destroyAllWindows()

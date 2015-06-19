import numpy as np
import cv2
import sys
import os

def spitframes(video_path, frame_output_folder):
	cap = cv2.VideoCapture(video_path)

	counter = 0
	while(cap.isOpened()):
		ret, frame = cap.read()
		if frame == None or frame.size == 0:
			break
		cv2.imwrite(os.path.join(frame_output_folder, str(counter) + '.jpg'), frame)
		counter += 1
	cap.release()
	return

video_folder = sys.argv[1]
output_folder = sys.argv[2]

for v in os.listdir(video_folder):
	new_videoname = '_'.join(v.split('.')[0].split(' '))
	print output_folder, new_videoname
	if not os.path.exists(os.path.join(output_folder, new_videoname)):	
		os.makedirs(os.path.join(output_folder, new_videoname))
	
	spitframes(os.path.join(video_folder,v ), os.path.join(output_folder, new_videoname))

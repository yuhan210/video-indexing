import cv2
import numpy


img = cv2.imread('/home/t-yuche/deep-video/data/frames/beyonce__drunk_in_love__red_couch_session_by_dan_henig_a1puW6igXcg/0.jpg', 1)
img_out = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
y,u,v = cv2.split(img_out)

print numpy.mean(y)

cv2.imshow('y', y)
cv2.waitKey(-1)

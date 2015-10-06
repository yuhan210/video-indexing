import cv2
import numpy as np
from PIL import Image
import imagehash

def getSobel(img, k_size = 3):

    ddepth = cv2.CV_16S
    scale = 1
    delta = 0

    cv2.GaussianBlur(img, (3,3), 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Gradient-x
    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize = k_size, scale = scale, delta = delta, borderType = cv2.BORDER_DEFAULT)

   
    #Gradient-y
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize = k_size, scale = scale, delta = delta, borderType = cv2.BORDER_DEFAULT)

    # converting back to uint8
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
   
    dst = cv2.addWeighted(abs_grad_x,0.5,abs_grad_y,0.5,0)
    #dst = cv2.add(abs_grad_x,abs_grad_y)

    return dst.mean()

def getIlluminance(img):

    img_out = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y,u,v = cv2.split(img_out) 
    ave_lum = np.mean(y)
    
    return ave_lum

def phash(a, b):
    a_hash = imagehash.phash(a)
    b_hash = imagehash.phash(b)

    return a_hash - b_hash

def dhash(a, b):
    a_hash = imagehash.dhash(a)
    b_hash = imagehash.dhash(b)

    return a_hash - b_hash

def ahash(a, b):
    a_hash = imagehash.average_hash(a)
    b_hash = imagehash.average_hash(b)
    
    return a_hash - b_hash

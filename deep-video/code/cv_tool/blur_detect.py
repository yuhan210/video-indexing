import cv2
import numpy as np
from matplotlib import pyplot as plt



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

    return dst
    

if __name__ == "__main__":

    img_path = 'C:/Users/t-yuche/Desktop/IMG_3175.JPG'
    img = cv2.imread(img_path)
    cv2.namedWindow("sobel 1", cv2.WINDOW_NORMAL)
    cv2.imshow('sobel 1', getSobel(img))
    cv2.resizeWindow('sobel 1', 640, 480)
    #print getSobel(img)

    img_path = 'C:/Users/t-yuche/Desktop/IMG_3178.JPG'
    img = cv2.imread(img_path)
    cv2.namedWindow("sobel 2", cv2.WINDOW_NORMAL);
    cv2.resizeWindow('sobel 2', 640, 480)
    cv2.imshow('sobel 2', getSobel(img))
    cv2.waitKey(-1)

    

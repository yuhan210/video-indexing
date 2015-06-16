import cv2

def play_video(video_path, speed):
    cap = cv2.VideoCapture(video_path) 
    while(cap.isOpened()):
        ret, frame = cap.read()
        if frame.width == 0 or frame.height == 0:
            break
        
        cv2.imshow('video', frame)
        cv2.waitKey(33/speed) 
    cap.release()
    cv2.destroyAllWindows()

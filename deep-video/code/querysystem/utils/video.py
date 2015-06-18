import cv2

def play_video(video_path, speed):
    video_name = video_path.split('/')[-1]
    print video_name
    cap = cv2.VideoCapture(video_path) 

    cv2.namedWindow( video_name, cv2.WINDOW_AUTOSIZE ); 
  
    counter = 0 
    while(cap.isOpened()):
        ret, frame = cap.read()
        
        if frame.shape[0] == 0 or frame.shape[1] == 0:
            break
        
        cv2.imshow(video_name, frame)
        cv2.waitKey(33/speed)
        counter += 1
        if counter == 200:
            break
    cap.release()
    cv2.destroyWindow(video_name)

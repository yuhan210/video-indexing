from tools.utils import *
from unidecode import unidecode
import sys
import cv2
import os

def draw_rects(img, rects, color):
    for x1, y1, w, h in rects:
        cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), color, 2)


if __name__ == "__main__":
    
    video_name = sys.argv[1]

    ocr_data = load_video_ocr('/home/t-yuche/frame-analysis/ocr-info', video_name)
    ped_data = load_video_peopledet('/home/t-yuche/frame-analysis/ped-info', video_name)
    face_data = load_video_opencvfd('/home/t-yuche/frame-analysis/face-info', video_name)
    rcnn_data = load_video_rcnn('/home/t-yuche/frame-analysis/rcnn-info', video_name)
    face_dlib_data = load_video_dlibfd('/home/t-yuche/frame-analysis/face-info', video_name)
    vgg_data = load_video_recog('/home/t-yuche/frame-analysis/recognition', video_name)
    caption_data = load_video_caption('/home/t-yuche/frame-analysis/caption', video_name)
    
    frame_folder = '/home/t-yuche/deep-video/data/frames/' + video_name
    
    for idx in xrange(len(ocr_data)):
        
        ocr = ocr_data[idx]
        ped = ped_data[idx]
        face = face_data[idx]
        rcnn = rcnn_data[idx]
        caption = caption_data[idx]
        vgg_recog = vgg_data[idx]
        dlib_data = face_dlib_data[idx]
        dlib_faces = [dlib_face[0] for dlib_face in face_dlib_data[idx]['faces']]
        
        img_name = face['img_name']
        img = cv2.imread(os.path.join(frame_folder, img_name))
        h, w, c = img.shape
        # draw faces
        draw_rects(img, face['faces'], (0,255,255))
        draw_rects(img, dlib_faces, (255, 255, 0))
        
        # draw peds
        # draw_rects(img, ped['peds'], (0, 255, 0))
        # draw ocr
        ocr_str = unidecode(ocr['ocr'])
        print ocr_str
        #cv2.putText(img, ocr_str, (30, h-30), cv2.FONT_HERSHEY_SIMPLEX, 1, 2, 255)
       
        # draw caption verbs
        verbs = getVerbTfFromCaps(caption['candidate']['text'])
        for idx, v in enumerate(verbs):
            render_str = '%s: %.4f' %(v, verbs[v]/(sum([verbs[key] for key in verbs]) * 1.0 ))
            cv2.putText(img, render_str, (w-400, 30 + 30 * idx), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
             
        # draw vgg recog 
        for idx, s in enumerate(vgg_recog['pred']['text']):
            render_str = '%s:%.4f' %(s, rcnn['pred']['conf'][idx]) 
            cv2.putText(img, render_str, (30, int(0.6 * h + 30 * idx)), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
   
 
        # draw rcnn detection        
        rcnn_str = ','.join(rcnn['pred']['text']) 
        for idx, s in enumerate(rcnn['pred']['text']):
            render_str = '%s:%.4f' %(s, rcnn['pred']['conf'][idx]) 
            cv2.putText(img, render_str, (30, 30 + 30 * idx), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)

        cv2.namedWindow(img_name)
        cv2.moveWindow(img_name, 30, 30)
        cv2.imshow(img_name, img)
        cv2.waitKey(500)
        cv2.destroyWindow(img_name)

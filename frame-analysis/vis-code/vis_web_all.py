import sys
import os
import json
from tools.utils import *
from unidecode import unidecode


def getRecogByImgname(recog_data, img_name):
    for recog in recog_data:
        if recog['img_path'].find('/' + img_name) >= 0:
            return recog

def getCaptionByImgname(caption_data, img_name):
    for caption in caption_data:
        if caption['img_path'].find('/' + img_name) >= 0:
            return caption

if len(sys.argv) != 2:
    print 'Usage', sys.argv[0], ' video_name'
    exit(-1)


video_name = sys.argv[1]
print video_name

##
ocr_data = load_video_ocr('/home/t-yuche/frame-analysis/ocr-info', video_name)
ped_data = load_video_peopledet('/home/t-yuche/frame-analysis/ped-info', video_name)
face_data = load_video_opencvfd('/home/t-yuche/frame-analysis/face-info', video_name)
rcnn_data = load_video_rcnn('/home/t-yuche/frame-analysis/rcnn-info', video_name)
face_dlib_data = load_video_dlibfd('/home/t-yuche/frame-analysis/face-info', video_name)
vgg_data = load_video_recog('/home/t-yuche/frame-analysis/recognition', video_name)
caption_data = load_video_caption('/home/t-yuche/frame-analysis/caption', video_name)


# read blur info
blur_json = os.path.join('/home/t-yuche/frame-analysis/blur-info', video_name + '_blur.json')
with open(blur_json) as blur_file:
    blur_data = json.load(blur_file)

blur_data = sorted(blur_data['img_blobs'], key=lambda x:int(x['img_name'].split('.')[0]))
#sort_blur_data = sorted(blur_data['img_blobs'], key=lambda x: x['blur_measure'] )

html = '<table border="1" style="width:100%">  \
        <tr> \
        <td> <b> Image </b> </td> \
        <td> <b> Caption beamsize 5 (CNN+RNN)</b> <br> <i>(log prob) caption </i> </td> \
        <td> <b> Object Recog. (CNN with VGG) </b> <br> <i> (conf) label </i></td> \
        <td> <b> Object Recog. (RCNN) </b> <br> <i> (conf) label </i></td> \
        <td> <b> Face detection </b> </td> \
        <td> <b> OCR </b> </td> \
        <td> <b> Blur Measure (Sobel operator).</b> Smaller -> blurred </td> \
        </tr>'

for idx in xrange((len(ocr_data))):
    
    ocr = ocr_data[idx]
    ped = ped_data[idx]
    face = face_data[idx]
    #
    rcnn = rcnn_data[idx]
    #
    caption = caption_data[idx]
    #
    vgg_recog = vgg_data[idx]
    dlib_data = face_dlib_data[idx]
    dlib_faces = [dlib_face[0] for dlib_face in face_dlib_data[idx]['faces']]
    blur = blur_data[idx]  
 
    img_name = face['img_name']
     
    if idx % 1 == 0:

        html += '<tr>'

        # caption
        verbs = getVerbTfFromCaps(caption['candidate']['text'])
        html += '<td><img src="%s" height=240 width=320><br>%s</td>' % ( './' + video_name + '/' + img_name, img_name)
        
        html += '<td> RNN: %f secs <br>' % (caption['rnn_time'])
        for idx, v in enumerate(verbs):
            render_str = '%s: %.4f <br>' %(v, verbs[v]/(sum([verbs[key] for key in verbs]) * 1.0 ))
            html += render_str
        html += '<br>'
        for i in xrange(len(caption['candidate']['text'])):
            html += '(%f) %s<br>' % (caption['candidate']['logprob'][i], caption['candidate']['text'][i])
        html += '</td>'

        # vgg
        html += '<td>'
        for i in xrange(len(vgg_recog['pred']['text'])):
            html += '(%f) %s<br>' % (-vgg_recog['pred']['conf'][i], vgg_recog['pred']['text'][i])
        html += '</td>'
       
        # rcnn 
        html += '<td>'
        for i in xrange(len(rcnn['pred']['text'])):
            html += '(%f) %s<br>' % (rcnn['pred']['conf'][i], rcnn['pred']['text'][i])
        html += '</td>'
        
        # face
        # a list of bbox
        html += '<td>'
        html += '(%s, %s)' % (str(len(face['faces'])), str(len(dlib_faces)))  
        html += '</td>'
       
        # ocr
        ocr_str = unidecode(ocr['ocr'])
        html += '<td>'
        html += '%s' % ocr_str 
        html += '</td>'
         
        # blur measure        
        html += '<td> %f </td>' % (blur['blur_measure'])
        html += '</tr>'
    
html += '</table>'
html_file = '/home/t-yuche/frame-analysis/tmp/' + video_name + '_results.html'
open(html_file, 'w').write(html)


import csv
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

def load_anno(anno_folder, video_name):
    files = os.listdir(os.path.join(anno_folder, video_name))
    files = sorted(files, key=lambda x: int(x.split('.')[0])) 
   
    ds = {}
    for f in files:
        with open(os.path.join(anno_folder, video_name, f)) as json_file:
            anno_data = json.load(json_file)  
            ds[f.split('.')[0] + '.jpg'] = anno_data['choices']
                     
    return ds

def load_amt(amtresults_folder):
    
    ds = {}
    for f in os.listdir(amtresults_folder):
        csv_file = open(os.path.join(amtresults_folder, f))
        csv_reader = csv.DictReader(csv_file, delimiter="\t")
        
        for row in csv_reader:
            if row['Answer.n_selections'] != None and len(row['Answer.n_selections']) > 0:
                video_name = row['Answer.video']
                frame_name = row['Answer.frame_name']
                selections = row['Answer.selections'].split(',')
                ds[video_name +  frame_name] = selections
        csv_file.close()
                 
    return ds            

        
def gen_html(video_name, max_choice = 15):

    ##
    anno_data = load_anno('/home/t-yuche/gt-labeling/frame-subsample/annos', video_name)
    amt_data = load_amt('/home/t-yuche/gt-labeling/process-amt-results/data')
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
            <td> <b> Suggested labels. </b> </td> \
            <td> <b> Turker output.</b> </td> \
            </tr>'

    # sort anno_data
    anno_keys = sorted(anno_data.keys(), key=lambda x: int(x.split('.')[0]))
    print anno_keys
    for idx, img_name in enumerate(anno_keys):
  
        print img_name 
        idx = int(img_name.split('.')[0])
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
        
            # suggested labels measure
            html += '<td>'
            for i in xrange(min(max_choice, len(anno_data[img_name]))):
                html += '%s <br>' % (anno_data[img_name][str(i)])
            html += '</td>'
    

            html += '<td>'
            if (video_name + img_name) in amt_data:
                for label in amt_data[video_name + img_name]:
                
                    if label == '_none':
                        html += 'None of the above <br>'
                        continue

                    label_idx = label.split('-')[0]
                    starting_label = label.split('-')[-1]
            
                    # composing output labels                
                    match = False
                    matched_label = []
                    for l in anno_data[img_name][label_idx].split('->'):
                        if l == starting_label:
                            match = True
                        if match:
                            matched_label += [l]
                    ## 
                    html += '%s <br>' % ' -> '.join(matched_label)

            html += '</td>'
            html += '</tr>'
                
   

    html += '</table>'
    html_file = '/home/t-yuche/gt-labeling/process-amt-results/vis-amt-html/' + video_name + '_amtresults.html'
    open(html_file, 'w').write(html)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print 'Usage', sys.argv[0], ' video_list'
        exit(-1)

    max_choice = 15
    video_list = sys.argv[1]
    for video_name in open(video_list).readlines():
        video_name = video_name.strip()
        print video_name
        gen_html(video_name)

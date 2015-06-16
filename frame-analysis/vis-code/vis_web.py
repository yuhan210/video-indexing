import sys
import os
import json



def getRecogByImgname(recog_data, img_name):
    for recog in recog_data:
        if recog['img_path'].find('/' + img_name) >= 0:
            return recog

def getCaptionByImgname(caption_data, img_name):
    for caption in caption_data:
        if caption['img_path'].find('/' + img_name) >= 0:
            return caption

if len(sys.argv) != 5:
    print 'Usage', sys.argv[0], ' video_folder(*/frames/video_name) caption_folder recognition_folder blur_folder'
    exit(-1)


video_folder = sys.argv[1]
caption_folder = sys.argv[2]
recog_folder = sys.argv[3]
blur_folder = sys.argv[4]

# read recognition results.
if len(video_folder.split('/')[-1]) == 0:
    video_name = video_folder.split('/')[-2]
else:
    video_name = video_folder.split('/')[-1]

if len(video_name) == 0:
    print 'Error! No video name'
    exit(-1)

print video_name
# read caption results.
'''
struct_json = os.path.join(summary_folder, video_name + '_1_caption.json')
with open(struct_json) as json_file:
   caption_data_1 = json.load(json_file)

caption_data_1 = sorted(caption_data_1['imgblobs'], key=lambda x:int(x['img_path'].split('/')[-1].split('.')[0]))
'''

##
struct_json = os.path.join(caption_folder, video_name + '_5_caption.json')
with open(struct_json) as json_file:
   caption_data_5 = json.load(json_file)

caption_data_5 = sorted(caption_data_5['imgblobs'], key=lambda x:int(x['img_path'].split('/')[-1].split('.')[0]))


##
'''
struct_json = os.path.join(summary_folder, video_name + '_7_caption.json')
with open(struct_json) as json_file:
   caption_data_7 = json.load(json_file)

caption_data_7 = sorted(caption_data_7['imgblobs'], key=lambda x:int(x['img_path'].split('/')[-1].split('.')[0]))
'''

##
'''
struct_json = os.path.join(summary_folder, video_name + '_20_caption.json')
with open(struct_json) as json_file:
   caption_data_20 = json.load(json_file)

caption_data_20 = sorted(caption_data_20['imgblobs'], key=lambda x:int(x['img_path'].split('/')[-1].split('.')[0]))
'''

# read recognition results
with open(os.path.join(recog_folder,  video_name + '_recog.json')) as json_file:
    recog_data = json.load(json_file)

recog_data = sorted(recog_data['imgblobs'], key=lambda x:int(x['img_path'].split('/')[-1].split('.')[0]))

# read blur info
blur_json = os.path.join(blur_folder, video_name + '_blur.json')
with open(blur_json) as blur_file:
    blur_data = json.load(blur_file)

#blur_data = sorted(blur_data['img_blobs'], key=lambda x:int(x['img_name'].split('.')[0]))
sort_blur_data = sorted(blur_data['img_blobs'], key=lambda x: x['blur_measure'] )

html = '<table border="1" style="width:100%">  \
        <tr> \
        <td> <b> Image </b> </td> \
        <td> <b> Caption beamsize 5 (CNN+RNN)</b> <br> <i>(log prob) caption </i> </td> \
        <td> <b>Obj. Recog. (CNN with VGG) sorted by confidence</b> <br> <i> (conf) label </i></td> \
        <td> <b>Blur Measure (Sobel operator).</b> Smaller -> blurred </td> \
        </tr>'

idx = 0
for img in sort_blur_data:
    img_name = img['img_name']
    if idx % 1 == 0:
        cur_recog = getRecogByImgname(recog_data, img_name)
        #cur_blur = blur_data[idx]
        cur_caption_5 = getCaptionByImgname(caption_data_5, img_name)
        html += '<tr>'
        html += '<td> %s <br>' % (img_name) 
        html += '<img src="%s" height=240 width=320></td>' % (os.path.join(os.path.abspath(video_folder), img['img_name'].split('/')[-1]))
        
        html += '<td> RNN: %f secs <br>' % (cur_caption_5['rnn_time'])
        for i in xrange(len(cur_caption_5['candidate']['text'])):
            html += '(%f) %s<br>' % (cur_caption_5['candidate']['logprob'][i], cur_caption_5['candidate']['text'][i])
        html += '</td>'
        
        html += '<td>'
        for i in xrange(len(cur_recog['pred']['text'])):
            html += '(%f) %s<br>' % (cur_recog['pred']['conf'][i], cur_recog['pred']['text'][i])
        html += '</td>'
        
        html += '<td> %f </td>' % (img['blur_measure'])
        html += '</tr>'
    idx += 1
html += '</table>'
html_file = '/home/t-yuche/frame-analysis/vis/' + video_name + '_sortblur_results.html'
open(html_file, 'w').write(html)


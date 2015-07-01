from tools.utils import *
from nltk.corpus import stopwords
import sys
import json


def removeStopWords(list_str):
    ws = []
    for s in list_str:
       ws.extend( [w for w in s.split(' ') if w not in stopwords.words('english')] )

    return ws

def genBow(rcnn_ws, vgg_ws, caption_ws):

    words = {}
    for w in rcnn_ws:
        if w not in words:
            words[w] = 5
        else:
            words[w] += 5

    for w in vgg_ws:
        if w not in words:
            words[w] = 5
        else:
            words[w] += 5

    for w in caption_ws:
        if w not in words:
            words[w] = 1
        else:
            words[w] += 1

    return words

def getSuggestedLabel(rcnn_data, vgg_data, caption_data, start_frame, end_frame):

    start_idx = int(start_frame.split('.')[0])
    end_idx = int(end_frame.split('.')[0])

    rcnn_data = filter(lambda x: int(x['image_path'].split('/')[-1].split('.')[0]) >= start_idx and int(x['image_path'].split('/')[-1].split('.')[0]) <= end_idx , rcnn_data) 
    vgg_data = filter(lambda x: int(x['img_path'].split('/')[-1].split('.')[0]) >= start_idx and int(x['img_path'].split('/')[-1].split('.')[0]) <= end_idx , vgg_data) 
    caption_data = filter(lambda x: int(x['img_path'].split('/')[-1].split('.')[0]) >= start_idx and int(x['img_path'].split('/')[-1].split('.')[0]) <= end_idx , caption_data) 
  
    labels = []
    range_bows = {}
    for idx in xrange(len(rcnn_data)):
        rcnn_ws = [w for w in rcnn_data[idx]['pred']['text'][:3] if w.find('background') < 0]
        vgg_ws = [w for w in vgg_data[idx]['pred']['text']]
        caption_ws = removeStopWords( caption_data[idx]['candidate']['text'] )

        bow = genBow(rcnn_ws, vgg_ws, caption_ws) 
        for w in bow:
            if w not in range_bows:
                range_bows[w] = 1
            else:
                range_bows[w] += 1

    range_bows = sorted(range_bows.items(), key=lambda x: x[1], reverse=True)    
    labels = [x for x in range_bows]
               
    return labels 
    

if __name__ == "__main__":

 
    video_list = [x.strip() for x in open('/home/t-yuche/deep-video/data/videos.txt').readlines()]

    # for each video
    for v in video_list:
      
         
        print v 
        templates = [x[:-1] for x in open('test.html').readlines()]
        rcnn_data = load_video_rcnn('/home/t-yuche/frame-analysis/rcnn-info', v)
        vgg_data = load_video_recog('/home/t-yuche/frame-analysis/recognition', v)
        caption_data = load_video_caption('/home/t-yuche/frame-analysis/caption', v)
        out_file = './vis/' + v + '_amtlabel.html'       
 
        keyframes = json.load(open('./keyframe-info/' + v + '_05.json'))['img_blobs']

        total_kframes = len(keyframes)
        total_frames = len(rcnn_data)
      
 
        selectrate_str = '%.4f (%d/%d)' % ((total_kframes/(total_frames * 1.0)), total_kframes, total_frames)
        body_html = '<br> \
                    Selection rate:' + selectrate_str + '<br> \
                    <table border="1" style="width:100%>" \
                    <tr> \
                    <td> <b> Image </b> </td> \
                    <td> <b> Timestamp </b> </td> \
                    <td> <b> Suggestions </b> </td> \
                    </tr>'

        # write table (process possible labels)  
        for idx, kf in enumerate(keyframes):
            
            kf_name = kf['key_frame']
            frame_idx = int(kf_name.split('.')[0])
            timestamp = '%s:%s' % (str(frame_idx/1800), str((frame_idx%1800)/30) ) 
            
            # get labels
            labels = [] 
            if idx == total_kframes - 1:
                labels = getSuggestedLabel(rcnn_data, vgg_data, caption_data, keyframes[idx]['key_frame'], rcnn_data[len(rcnn_data)-1]['image_path'].split('/')[-1])
                
            else:
                labels = getSuggestedLabel(rcnn_data, vgg_data, caption_data, keyframes[idx]['key_frame'], keyframes[idx+1]['key_frame'])
           
            body_html += '<tr>'
            body_html += '<td><img src=%s height=240 width=320><br>%s</td>' % ('./' + v + '/' + kf_name, kf_name) 
            body_html += '<td> %s </td>' % (timestamp)
            
            body_html += '<td>'
            for label in labels[:10]:
                body_html += '%s <br>' % str(label[0])
            body_html += '</td>' 
            body_html += '</tr>' 
    
        body_html += '</table>\n'
    
        # write index file (luckyframe.html)
        output_html = templates
        output_html.insert(38, body_html)
        output_html[19]  = '\t\tvideoId:"' + v[-11:] + '",'
        out_fh = open(out_file, 'w') 
        out_fh.write('\n'.join(output_html))
        out_fh.close()

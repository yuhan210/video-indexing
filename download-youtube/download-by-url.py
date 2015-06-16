from bs4 import BeautifulSoup
import urllib2
import pafy
import sys
import json
import re
import os

def getWatchUrls(query):

    query_url = "https://www.youtube.com/results?search_query=" + query
    watch_urls = []

    try:
        html_doc = urllib2.urlopen(query_url).readlines()

    except urllib2.URLError, e:
        # For Python 2.7
        print 'URLError %r' % e
        exit(-1)

    except socket.timeout, e:
        # For Python 2.7
        print 'Timeout %r' % e
        exit(-1)

    for line in html_doc:
        if line.find('</div><div class="yt-lockup-content">') >= 0:
            watch_url = line.split('a href')[1].split('class=')[0].split('"')[1]
            watch_urls += [watch_url]
            
    return watch_urls


def createVideoMeatadata(video_obj, new_filename): 
    
    v = {}
    v['otitle'] = video_obj.title
    v['videoid'] = video_obj.videoid
    v['ftitle'] = new_filename
    v['videoname'] = new_filename + '.mp4'
    v['author'] = video_obj.author 
    v['category'] = video_obj.category
    v['description'] = video_obj.description
    v['likes'] = video_obj.likes
    v['dislikes'] = video_obj.dislikes
    v['duration'] = video_obj.duration
    v['keywords'] = video_obj.keywords
    v['length'] = video_obj.length
    v['published'] = video_obj.published
    v['rating'] = video_obj.rating
    v['viewcount'] = video_obj.viewcount

    return v

## get new video name without extension
def getNewVideoName(video):

    filename = video.title
    videoname = filename.split('.mp4')[0]
    nstr = re.sub(r'[?|$|.|!]',r'', videoname)
    nestr = nestr = re.sub(r'[^a-zA-Z0-9 ]',r'',nstr)
    new_videoname = '_'.join([str(x).lower() for x in nestr.split(' ')]) + '_' + video.videoid
    
    '''
    # check if there's an existing video with the same name
    tmp_videoname = new_videoname
    count = 1
    while (os.path.exists(os.path.join(video_folder, tmp_videoname + '.mp4'))):
        tmp_videoname = new_videoname + "_" + str(count)
        count += 1

    new_videoname = tmp_videoname
    '''

    return new_videoname

def downloadVideo(url, video_folder="/home/t-yuche/deep-video/data/videos", video_meta_folder="/home/t-yuche/deep-video/data/video-info"):
    
    video = pafy.new(url)   

    new_filename = getNewVideoName(video)  
    
    best_video = video.getbest(preftype="mp4")
    best_video.download(quiet=False, filepath=os.path.join(video_folder,  new_filename + ".mp4"))

    # write video metadata in the json format
    v_info = createVideoMeatadata(video, new_filename)
    with open(os.path.join(video_meta_folder, new_filename + '.json'), 'w') as fh:
        json.dump(v_info, fh)


if __name__ == "__main__":

    url = (raw_input('$ url:'))
    
    #query = '+'.join(input_str.split(' '))
    #https://www.youtube.com/results?search_query=basketball        
    #watch_urls = getWatchUrls(query)
    
    # get top-10 urls
    downloadVideo(url) 
            

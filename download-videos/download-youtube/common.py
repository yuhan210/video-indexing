import pafy
import json
import os
import re


## get new video name without extension
def getNewVideoName(video):

    filename = video.title
    videoname = filename.split('.mp4')[0]
    nstr = re.sub(r'[?|$|.|!]',r'', videoname)
    nestr = nestr = re.sub(r'[^a-zA-Z0-9 ]',r'',nstr)
    new_videoname = '_'.join([str(x).lower() for x in nestr.split(' ')]) + '_' + video.videoid
    
    return new_videoname

    
def downloadVideo(video, video_folder="/home/t-yuche/remote-disk/videos", video_meta_folder="/home/t-yuche/remote-disk/video-info"):

    new_filename = getNewVideoName(video)  
    print 'Downdloading %s' % new_filename
    # write video metadata in the json format
    v_info = createVideoMeatadata(video, new_filename)
    with open(os.path.join(video_meta_folder, new_filename + '.json'), 'w') as fh:
        json.dump(v_info, fh)
    
    best_video = video.getbest(preftype="mp4")
    best_video.download(quiet=False, filepath=os.path.join(video_folder,  new_filename + ".mp4"))

    return v_info

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
    
    best_video = video_obj.getbest(preftype="mp4")
    v['videoquality'] = best_video.bitrate
    v['videores'] = best_video.resolution
    

    return v



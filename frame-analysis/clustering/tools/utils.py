import os
import json

def load_video_recog(recog_folder, video_name):
    
    file_pref = os.path.join(recog_folder, video_name)
    
    # load recognition
    with open(file_pref + '_recog.json') as json_file:
        recog_data = json.load(json_file)

    recog_data = sorted(recog_data['imgblobs'], key=lambda x: int(x['img_path'].split('/')[-1].split('.')[0]))
    
    return recog_data



def load_video_caption(caption_folder, video_name):

    file_pref = os.path.join(caption_folder, video_name)

    # load caption
    with open(file_pref + '_5_caption.json') as json_file:
        caption_data = json.load(json_file)

    # sort caption results based on frame number
    caption_data = sorted(caption_data['imgblobs'], key=lambda x:int(x['img_path'].split('/')[-1].split('.')[0]))

    return caption_data


def load_video_blur(blur_folder, video_name):

    file_pref = os.path.join(BLUR_folder, video_name)

    # load blurinfo
    with open(file_pref + '_blur.json') as json_file:
        blur_data = json.load(json_file)

    blur_data = sorted(blur_data['img_blobs'], key=lambda x:int(x['img_name'].split('.')[0]))

    return blur_data



def load_video_summary(summary_folder, video_name):

    file_pref = os.path.join(summary_folder, video_name)

    # load caption
    with open(file_pref + '_5_caption.json') as json_file:
        caption_data = json.load(json_file)

    # sort caption results based on frame number
    caption_data = sorted(caption_data['imgblobs'], key=lambda x:int(x['img_path'].split('/')[-1].split('.')[0]))

    # load recognition
    with open(file_pref + '_recog.json') as json_file:
        recog_data = json.load(json_file)

    recog_data = sorted(recog_data['imgblobs'], key=lambda x: int(x['img_path'].split('/')[-1].split('.')[0]))

    # load blurinfo
    with open(file_pref + '_blur.json') as json_file:
        blur_data = json.load(json_file)

    blur_data = sorted(blur_data['img_blobs'], key=lambda x:int(x['img_name'].split('.')[0]))

    return caption_data, recog_data, blur_data


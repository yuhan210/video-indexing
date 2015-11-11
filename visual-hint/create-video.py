from utils import *
import cv2


if __name__ == "__main__":

    videos = open('/mnt/video_list.txt').read().split()

    for video_name in videos:
        fps, w, h = get_video_fps(video_name + '.mp4')
        start_idx = 0
        end_idx = 30 * 5
        
        out_video_name = video_name + '_' + str(start_idx) + '_' + str(end_idx) + '.mp4'

        outstr = 'ffmpeg -framerate ' + str(fps) + ' -start_number ' + str(start_idx) + ' -i /mnt/frames/' + video_name + '/%d.jpg -vframes 150 -s 480x360 -vcodec mpeg4 ' + out_video_name 

        print outstr
        break

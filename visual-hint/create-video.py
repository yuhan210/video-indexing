from utils import *
import cv2


if __name__ == "__main__":

    lines = open('/home/t-yuche/visual-hint/video-seg-info/single_obj.txt').readlines()

    for line in lines:
        line =  line.strip()
        video_name = line.split(',')[0]
        start_idx = int(line.split(',')[1])
        end_idx = int(line.split(',')[2])
        n_obj = int(line.split(',')[4])
        
        n_shownf = int(line.split(',')[-1])
        n_objf = int(line.split(',')[-2])
        fps, w, h = get_video_fps(video_name + '.mp4')
        
        out_video_name = video_name + '_' + str(start_idx) + '_' + str(end_idx) + '.mp4'

        if n_objf/(n_shownf * 1.0) > 0.025 and n_objf > 2 and n_obj > 5:
            outstr = '~/ffmpeg/ffmpeg -framerate ' + str(fps) + ' -start_number ' + str(start_idx) + ' -i /mnt/frames/' + video_name + '/%d.jpg -vframes 150 -s 560x420 -vcodec libx264 -pix_fmt yuv420p ' + out_video_name 

            print outstr

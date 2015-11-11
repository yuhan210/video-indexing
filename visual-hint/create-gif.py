from utils import *


if __name__ == "__main__":

    videos = open('/mnt/video_list.txt').read().split()
    for video_name in videos:
        fps, w, h = get_video_fps(video_name + '.mp4')
        start_idx = 0
        end_idx = 30 * 5
        outgifname = video_name + '_' + str(start_idx) + '_' + str(end_idx) + '.gif'
        outstr = 'convert -fuzz 1% +dither -delay 1x' + str(fps) + ' `seq -f ' '/mnt/frames/' + video_name + '/%g.jpg ' + str(start_idx) + ' ' + str(end_idx) + '` -coalesce -layers OptimizeTransparency ' + outgifname 
       
        outreize_str = 'convert -size '+ str(w) +  'x' + str(h) +' ' + outgifname  + ' -resize 480x360 ' + outgifname

        print outstr
        print outreize_str
        break


import sys


video_list = sys.argv[1]
videos = [x.strip() for x in open(video_list).readlines()]


# ffprobe -show_frames -print_format json -show_streams -select_streams v ~/deep-video/data/videos/12\ year\ old\ Catcher\ celebrates\ a\ strikeout\ and\ a\ win...\ is\ ejected\!.mp4 > 'out.json'

for v in videos:
    print 'ffprobe -show_frames -show_data -show_packets -print_format json -show_streams -select_streams v /mnt/videos/' + v + '.mp4 > /mnt/tags/video-encoding-info/' + v + '_encoding.json\n\n'




import sys


video_list = sys.argv[1]
videos = [x.strip() for x in open(video_list).readlines()]


# ffprobe -show_frames -print_format json -show_streams -select_streams v ~/deep-video/data/videos/12\ year\ old\ Catcher\ celebrates\ a\ strikeout\ and\ a\ win...\ is\ ejected\!.mp4 > 'out.json'

for v in videos:
    print 'ffprobe -show_frames -print_format json -show_streams -select_streams v /home/t-yuche/deep-video/data/videos/' + v + '.mp4 > /home/t-yuche/deep-video/frame-analysis/video-summary/' + v + '_encoding.json\n\n'




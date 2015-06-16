import sys

video_list = sys.argv[1]
output_sh_filename = 'batch_web_vis.sh'
# python rewrite_web.py ./frames/GoPro-_Airplane_Waterskiing/ ./video_summary/
fh_out = open(output_sh_filename, 'w')
for v in open(video_list).readlines():
    v = v.strip()
    fh_out.write('python /home/t-yuche/frame-analysis/vis-code/vis_web.py /home/t-yuche/deep-video/data/frames/' + v + ' /home/t-yuche/frame-analysis/caption/ /home/t-yuche/frame-analysis/recognition /home/t-yuche/frame-analysis/blur-measure\n\n')

fh_out.close()

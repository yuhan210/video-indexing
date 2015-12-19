import os

if __name__ == "__main__":

    VIDEO_LIST = '/mnt/video_list.txt'
    videos = open(VIDEO_LIST).read().split()

    log_folder = '/mnt/tags/edgebox-all'
    tv_n = len(videos)
    pv_n = 0
    tf_n = 0
    pf_n = 0
    for vid, video_name in enumerate(videos):
        #print vid, video_name
        vpf_n = 0 
        if os.path.exists(os.path.join(log_folder, video_name)):
            vpf_n = len(os.listdir(os.path.join(log_folder, video_name)))
        vtf_n = len(os.listdir(os.path.join('/mnt/frames', video_name)))
        
        tf_n += vtf_n
        pf_n += vpf_n
        
        if vpf_n == vtf_n:
            pv_n += 1

    print 'Processed videos:', pv_n/(tv_n * 1.0), '(', pv_n, '/', tv_n, ')'
    print 'Processed frames:', pf_n/(tf_n * 1.0), '(', pf_n, '/', tf_n, ')'

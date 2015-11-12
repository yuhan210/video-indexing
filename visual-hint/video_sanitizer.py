from utils import *


if __name__ == "__main__":

    lines = open('/home/t-yuche/visual-hint/video-seg-info/single_obj_tmp').readlines()
    outfh = open('/home/t-yuche/visual-hint/video-seg-info/single_obj.txt', 'w')

    for line in lines:
        line =  line.strip()
        video_name = line.split(',')[0]
        start_fid = int(line.split(',')[1])
        end_fid = int(line.split(',')[2])
        dummy, label_dict = load_video_processed_turker(video_name)
         
        obj_count = 0
        labeled_count = 0 
        for fid in xrange(start_fid, end_fid + 1): 
            frame_name = str(fid) + '.jpg'
            if frame_name in label_dict:
                gt_labels = label_dict[frame_name] 
                labeled_count += 1 
                if 'dog' in gt_labels: 
                    obj_count += 1

        if obj_count > 0:
            outfh.write(line.strip() + ',' + str(obj_count) + ',' + str(labeled_count) +  '\n')
            outfh.flush()

    outfh.close() 

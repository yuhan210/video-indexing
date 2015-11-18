import random
#feature_str = video_name + ','  + str(fid) + ',' + enc['type'] + ',' +  str(w) + ',' + str(h) + ',' + str(enc['size']) + ',' + str(mv_features[fid][0]) + ',' + str(mv_features[fid][1]) + ',' + str(mv_features[fid][2]) + ',' + str(mv_features[fid][3]) +  ',' + str(cv['sobel'][0]) + ',' + str(cv['illu'][0]) + ',' +str(btf_cv['framediff'][0]) + ',' + str(btf_cv['phash'][0]) + ','+ str(btf_cv['colorhist'][0]) +','+ str(btf_cv['siftmatch'][0]) +','+ str(btf_cv['surftime'][0]) +','+ str(label)

#100_mile_wilderness_sled_dog_race_Qv4I_MDX7ws,1,P,1280,720,15463,4598,0.00170825660936,0.00972222222222,0.0,38.7681770833,140.846015625,0.0183333333333,0,0.993986833909,0.782352941176,0.8203125,0
if __name__ == "__main__":

    INPUT_TRAIN_DATA = "/home/t-yuche/admission-control/train/train.data"
    videos = open('/mnt/video_list.txt').read().split()
    lines = [x.strip() for x in open(INPUT_TRAIN_DATA).readlines()]

    train_data = {}
    for video in videos:
        train_data[video] = {'0': [], '1': []}
    
    prev_fid = 0 
    prev_video_name = ''
    for line in lines: 
        segs = line.split(',')
        video_name = segs[0]
        if video_name != prev_video_name:
            prev_fid = 0
            prev_video_name = video_name
        fid = segs[1]
        if segs[2] == 'P':
            enctype = '0'
        else:
            enctype = '1'
        w = segs[3]
        h = segs[4]
                
         
        encsize = segs[5]
        
        mvsize = segs[6]
        meanmv = segs[7]
        maxmv = segs[8]
        minmv = segs[9]
        sobel = segs[10]
        illu = segs[11] 
        framediff = segs[12]
        phash = segs[13]
        colorhist = segs[14]
        siftmatch = segs[15]
        if segs[15] == '-1':
            siftmatch = '0'
        surfmatch = segs[16]
        if segs[16] == '-1':
            surfmatch = '0'
        label = segs[17]
        encsize_norm = float(encsize)/ (float(w)*float(h))
        
        dist_from_prevfid = (int(fid) - int(prev_fid))
        if int(label) == 1:
            prev_fid = int(fid)
 
        feature_tuple = [enctype, w, h, encsize, mvsize, meanmv, maxmv, minmv, sobel, illu, framediff, phash, colorhist, siftmatch, surfmatch, dist_from_prevfid] 

        train_data[video_name][label] += [feature_tuple]
        

    print 'Finish parsing train.data'
    
    outputfh = open('svm_train_all.data', 'w')
    for vid, video_name in enumerate(videos):
        print 'Write feature for video:', vid, video_name
        n_positive_sample = len(train_data[video_name]['1'])
        selected_pos_samples = train_data[video_name]['1']
        selected_neg_samples = train_data[video_name]['0']
        #selected_pos_samples = random.sample(train_data[video_name]['1'], n_positive_sample/2)
        #selected_neg_samples = random.sample(train_data[video_name]['0'], n_positive_sample)
       
        for sample in selected_pos_samples:
            # write sample
            output_str = '1 '
            for fid, feature in enumerate(sample):
                output_str += str(fid+1) + ':' + str(feature) + ' ' 
            output_str += '\n'
            outputfh.write(output_str) 
          
        for sample in selected_neg_samples:
            # write sample
            output_str = '0 '
            for fid, feature in enumerate(sample):
                output_str += str(fid+1) + ':' + str(feature) + ' ' 
            output_str += '\n' 
            outputfh.write(output_str) 

        outputfh.flush()
    outputfh.close() 

import os
import sys
import json

##python extract_features.py --caffe ~/caffe/ --model_def ./deploy_features.prototxt --model ~/caffe/models/vgg_ilsvrc_16/VGG_ILSVRC_16_layers.caffemodel --files ~/neuraltalk/example_images/tasks.txt --out tmp


def loadKeyFrames(video_name):

    KEYFRAME_FOLDER = '/home/t-yuche/gt-labeling/frame-subsample/keyframe-info'
    keyframe_file = os.path.join(KEYFRAME_FOLDER, video_name + '_uniform.json')

    with open(keyframe_file) as json_file:
        keyframes = json.load(json_file)

    return keyframes


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print 'Usage', sys.argv[0], 'frame_folder(*/frame)'
        exit(-1)

    video_folder = sys.argv[1]

    output_file = 'gen_feature_caption.sh'
    fh_out = open(output_file, 'w')
    
    print os.listdir(video_folder)
    for v in os.listdir(video_folder):
        video_path = os.path.join(video_folder, v)
        if not os.path.isdir(video_path):
            continue
        
        print video_path

        # generate keyframe-tasks.txt
        # load key frames         
        keyframes = loadKeyFrames(v)
        img_names = [ os.path.join('/mnt/frames', v, x['key_frame']) for x in keyframes['img_blobs'] ]
        
        # load previously processed tasks         
        task_filename = os.path.join('/home/t-yuche/neuraltalk/python_features/tool/task', v + '_tasks.txt')
        processed_tasks = open(task_filename, 'r').read().splitlines()
       
        # write to a new task file
        new_task_filename = os.path.join('/home/t-yuche/neuraltalk/python_features/tool/keyframe-task', v + '_tasks.txt')
        fh_task_out = open(new_task_filename, 'w') 

        for f in img_names:
            if f in processed_tasks:
                continue
            fh_task_out.write(f + '\n')
            	
        
        fh_task_out.close()	
        # append to run script
        fh_out.write('python /home/t-yuche/neuraltalk/python_features/extract_features.py  --gpu --caffe ~/caffe/ --model_def /home/t-yuche/neuraltalk/python_features/deploy_features.prototxt --model /home/t-yuche/caffe/models/vgg_ilsvrc_16/VGG_ILSVRC_16_layers.caffemodel --files ' +  new_task_filename + ' --out /home/t-yuche/neuraltalk/python_features/tool/pickle-data/' + v + '.pickle\n')

        fh_out.write('python /home/t-yuche/neuraltalk/predict_on_keyframes.py /home/t-yuche/neuraltalk/models/flickr8k_cnn_lstm_v1.p -r ' + video_path +  ' -t ' + new_task_filename  + ' -f  /home/t-yuche/neuraltalk/python_features/tool/pickle-data/' + v + '.pickle ' + '-of  /mnt/tags/fei-caption-keyframe/' + v + '_5_caption.json &\n\n')

    fh_out.close()


	## prediction script
	##python predict_on_images.py lstm_model.p -r example_images
		

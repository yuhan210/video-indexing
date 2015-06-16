import os
import sys


##python extract_features.py --caffe ~/caffe/ --model_def ./deploy_features.prototxt --model ~/caffe/models/vgg_ilsvrc_16/VGG_ILSVRC_16_layers.caffemodel --files ~/neuraltalk/example_images/tasks.txt --out tmp

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print 'Usage', sys.argv[0], 'frame_folder(-/frame)'
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
        
        # generate tasks.txt
        task_filename = os.path.abspath(os.path.join('./tmp' , v + '_tasks.txt'))
        fh_task_out = open(task_filename, 'w')
	
        for f in os.listdir(video_path):
            fh_task_out.write(os.path.abspath(os.path.join(video_path, f)) + '\n')
        
        fh_task_out.close()	
        # append to run script
        fh_out.write('python /home/t-yuche/neuraltalk/python_features/extract_features.py  --gpu --caffe ~/caffe/ --model_def /home/t-yuche/neuraltalk/python_features/deploy_features.prototxt --model /home/t-yuche/caffe/models/vgg_ilsvrc_16/VGG_ILSVRC_16_layers.caffemodel --files ' +  task_filename + ' --out /home/t-yuche/neuraltalk/python_features/caption/' + v + '.pickle\n')

        fh_out.write('python /home/t-yuche/neuraltalk/predict_on_images.py /home/t-yuche/neuraltalk/models/flickr8k_cnn_lstm_v1.p -r ' + video_path +  ' -t ' + task_filename  + ' -f  /home/t-yuche/neuraltalk/python_features/caption/' + v + '.pickle ' + '-of  /home/t-yuche/neuraltalk/python_features/caption/' + v + '_5_caption.json\n\n')

    fh_out.close()




	## prediction script
	##python predict_on_images.py lstm_model.p -r example_images
		

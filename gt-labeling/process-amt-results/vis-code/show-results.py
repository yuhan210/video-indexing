import csv
import sys
import os
import cv2

if __name__ == "__main__":


    amtresults_folder = sys.argv[1]
    frame_folder = "/home/t-yuche/deep-video/data/frames" 
    

    for f in os.listdir(amtresults_folder):
        csv_file = open(os.path.join(amtresults_folder,f))
        #csv_reader = csv.reader(csv_file, delimiter= "\t")
        csv_reader = csv.DictReader(csv_file, delimiter='\t')
        for row in csv_reader:
            if row['Answer.n_selections'] != None and len(row['Answer.n_selections']) > 0:
                video_name = row['Answer.video']
                frame_name = row['Answer.frame_name']
                frame_path = os.path.join(frame_folder, video_name, frame_name)
                print frame_path
                img = cv2.imread(os.path.join(frame_folder, video_name, frame_name)) 
                selections = [x.split('-')[-1] for x in row['Answer.selections'].split(',')]
                
                render_str = ','.join(selections)
                cv2.putText(img, render_str, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
 
                cv2.imshow('amt', img)
                cv2.waitKey(1000)
        csv_file.close()

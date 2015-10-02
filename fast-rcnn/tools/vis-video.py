import matplotlib.pyplot as plt
import numpy as np
import json, os, cv2, sys


def vis_detections(im_path, bbx_objs, thresh=0.5):
    """Draw detected bounding boxes."""
    im = cv2.imread(im_path)
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for bbx_obj in bbx_objs:
        
        score = bbx_obj['score']            
        if score < thresh:
            continue
        bbox = bbx_obj['bbox']
        cls = bbx_obj['class']

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(cls, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.show()

def load_video_rcnn_bbx(rcnn_bbx_folder, video_name):

    file_pref = os.path.join(rcnn_bbx_folder, video_name)    

    # load rcnn bbx
    with open(file_pref + '_rcnnbbx.json') as json_file:
        rcnn_bbx_data = json.load(json_file)

    rcnn_bbx_data = sorted(rcnn_bbx_data['imgblobs'], key=lambda x: int(x['img_path'].split('/')[-1].split('.')[0]))

    return rcnn_bbx_data

if __name__ == "__main__":

    video_name = sys.argv[1]
    start_fid = int(sys.argv[2])
    #rcnn_data = load_video_rcnn_bbx('/mnt/tags/rcnn-bbx-tmp', 'dog_fight_pit_bull_owner_sues_family_for_1m_dollars_after_her_dogs_killed_their_beagle_sYJf_m0qDiw')
    #rcnn_data = load_video_rcnn_bbx('/mnt/tags/rcnn-bbx-tmp', 'dog_fight_pit_bull_owner_sues_family_for_1m_dollars_after_her_dogs_killed_their_beagle_sYJf_m0qDiw')
    rcnn_data = load_video_rcnn_bbx('/mnt/tags/rcnn-bbx-tmp', video_name)

    for fid, img_obj in enumerate(rcnn_data):
        if fid < start_fid:
            continue
        im_path = img_obj['img_path']
        vis_detections(im_path, img_obj['pred'], 0.4)
        
    

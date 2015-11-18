

def load_train_data(train_file):

    with open(train_file) as fh:
        samples = []
        pos_count = 0
        neg_count = 0
        for line in fh.readlines():
            line = line.strip()
            segs = line.split(' ')

            label = int(segs[0])
            if label == 1:
                pos_count += 1
            else:
                neg_count += 1

            sample = [label] # sample = [label, feature_1, feature_2, ...]
            for idx in xrange(1, len(segs)):
                fid = int(segs[idx].split(':')[0])
                v = float(segs[idx].split(':')[1])
                sample += [v]

            samples += [sample]

    return samples, pos_count, neg_count


def load_range_file(scale_file = './range'):
  
    scale_value = {}
    with open(scale_file) as f:
        for line in f.readlines():
            line = line.strip()
            segs = line.split(' ') 
            if len(segs) == 3:
                idx = int(segs[0])
                min_v = float(segs[1])
                max_v = float(segs[2])
                scale_value[idx] = (min_v, max_v)

    return scale_value


def load_line(line):
    line = line.strip()
    segs = line.split(' ')

    label = int(segs[0])

    sample = [label] # sample = [label, feature_1, feature_2, ...]
    for idx in xrange(1, len(segs)):
        fid = int(segs[idx].split(':')[0])
        v = float(segs[idx].split(':')[1])
        sample += [v]
    return sample

def convert_sample_to_dict(sample):
    feature = {}
    for fid, f in enumerate(sample):
        feature[fid] = f

    return feature

def scale_feature(sample, range_value):
    # sample = {(0):label, 1:feature_1, 2:feature_2, 3: feature_3...}

    upper = 1.0
    lower = -1.0

    for fid in sample:
        value = sample[fid]
        if fid in range_value.keys():
            scale_value = range_value[fid]
            if value <= scale_value[0]:
                value = lower
            elif value >= scale_value[1]:
                value = upper
            else:
                value = lower + (upper - lower) * (value - scale_value[0])/(scale_value[1] - scale_value[0])
        sample[fid] = value

    return sample


def feature_tostring(feature):

    output_str = str(feature[0]) 
    for fid in xrange(1, len(feature.keys())):

        output_str += ' ' + str(fid) + ':' + str(feature[fid])

    return output_str

'''
a = convert_sample_to_dict(load_line('1 1:0 2:1280 3:720 4:16988 5:4081 6:0.0161238227069 7:0.0475073093791 8:0.0 9:46.9089453125 10:104.912591146 11:0.0793424479167 12:35 13:0.70227203702 14:0.0 15:0.0234375 16:192'))

scale_value = load_range_file()
f =  scale_feature(a, scale_value)
print f
print feature_tostring(f)
'''

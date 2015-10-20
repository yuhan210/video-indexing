import numpy as np

if __name__ == "__main__":

    log_file = open('./sample_log.txt')
    lines = log_file.readlines()
    
    detailed_saves = []
    for idx, line in enumerate(lines):
        if idx % 4 == 0 and idx + 1 < len(lines):
            save = float(lines[idx + 1].split(':')[1])
            print save
            detailed_saves += [save] 

    s= '{:.2f}%'.format(np.mean(detailed_saves) * 100)
    print s

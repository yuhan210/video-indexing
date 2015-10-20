import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
try:
    plt.style.use('ggplot')
except:
    pass
import os

if __name__ == "__main__":

    winsizes = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 100]
    diffs = {}
    for winsize in winsizes:
        diffs[winsize]= []
        for f in os.listdir('./log'):
            #print winsize, f.split('.')[0]
            file_path = os.path.join('./log', f) 
            with open(file_path) as fh:
                exec_time = pickle.load(fh)

            batch_num = len(exec_time)/winsize + 1    
            for b in xrange(batch_num - 1):        
                cur_win = exec_time[b * winsize: (b+1) * winsize]  
                nex_win = exec_time[(b+1) * winsize: min(len(exec_time), (b+2) * winsize)]

                # compare two consecutive windows
                diff = abs(np.mean(cur_win) - np.mean(nex_win))
                diffs[winsize] += [diff] 

    with open('winsize.pickle', 'wb') as fh:
        pickle.dump(diffs, fh)

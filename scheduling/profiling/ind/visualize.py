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

    for f in os.listdir('./log'):
        file_path = os.path.join('./log', f) 
        with open(file_path) as fh:
            exec_time = pickle.load(fh)

        plt.plot(range(len(exec_time)), exec_time, 'o')
        plt.ylim([0, 1.2])
        plt.xlabel('Frame ID')
        plt.ylabel('Object detection time (sec)')
        plt.title(f)
        plt.show()         

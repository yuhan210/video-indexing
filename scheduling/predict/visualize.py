import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
try:
    plt.style.use('ggplot')
except:
    pass


if __name__ == "__main__":

   
    
 
    with open('rcnn_processing_time.pickle') as f:
        rcnn = np.sort(pickle.load(f))
    with open('msr_processing_time.pickle') as f:
        msr = np.sort(pickle.load(f))
    with open('obj_detection_processing_time.pickle') as f:
        edgebox = pickle.load(f)


    reso_n = sorted([len(edgebox[reso]) for reso in edgebox], reverse = True)
    
    plt.figure(1)
    plt.title('RCNN')
    plt.plot(rcnn, np.arange(len(rcnn))/float(len(rcnn)))
    plt.xlim([0, 0.4])
    plt.ylabel('CDF') 
    plt.xlabel('Execution time (sec)')
    plt.figure(2)
    plt.title('MSR captioning')
    plt.xlim([0, 0.5])
    plt.plot(msr, np.arange(len(msr))/float(len(msr)))
    plt.ylabel('CDF') 
    plt.xlabel('Execution time (sec)')
   
    counter = 0 
    for rid, reso in enumerate(edgebox):
        if len(edgebox[reso]) != reso_n[0] and len(edgebox[reso]) != reso_n[1] and len(edgebox[reso]) != reso_n[2]:
            continue
        plt.figure(counter + 3)
        data = edgebox[reso]
        data = np.sort(data)
        plt.title('Bounding Box Detection\n Image Resolution:' + reso)
        if reso == '480x360':
            plt.xlim([0, 1])
        elif reso == '640x360':
            plt.xlim([0, 1.2])
        else:
            plt.xlim([0, 5])
        
        plt.plot(data, np.arange(len(data))/float(len(data)))
        plt.ylabel('CDF') 
        plt.xlabel('Executon Time (secs)')
        counter += 1 
        

    plt.show() 

import random
import sys
import matplotlib
import matplotlib.pyplot as plt
try:
    plt.style.use('ggplot')
except:
    pass

if __name__ == "__main__":

    if sys.argv[1] == 'v':
        # plot data
        good_points = []
        bad_points = []
        with open('svm_train.data') as svm_fh:
            for line in svm_fh:
                line = line.strip()
                label = int(line.split(' ')[0])
                if label == 1:
                    good_points += [[float(x.split(':')[-1]) for x in line.split(' ')[1:]]]
                else:
                    bad_points += [[float(x.split(':')[-1]) for x in line.split(' ')[1:]]]
        plt.subplot(2, 1, 1)
        plt.scatter([x[0] for x in good_points], [x[1] for x in good_points], color='r', alpha = 0.7, label = 'Selected word')  
        plt.scatter([x[0] for x in bad_points], [x[1] for x in bad_points], color='b', alpha = 0.7, label = 'Unselected word') 
        plt.xlabel('RCNN')
        plt.ylabel('VGG')
        
        plt.subplot(2, 1, 2)
        plt.scatter([x[2] for x in good_points], [x[3] for x in good_points], color='r', alpha = 0.7, label = 'Selected word')  
        plt.scatter([x[2] for x in bad_points], [x[3] for x in bad_points], color='b', alpha = 0.7, label = 'Unselected word') 
        plt.xlabel('FEI')
        plt.ylabel('MSR')
        plt.legend()
        plt.show()

        exit(0)
    good_samples = []
    bad_samples = []

        
    # making each class to have the same number of sample points  
    with open('train_log.txt') as train_fh:
        for line in train_fh:
            line = line.strip()
            feature_start_idx = len(line.split()) - 5
            features = [float(x) for x in line.split()[feature_start_idx:-1]]
            label = int(line.split()[-1])
            
            if label == 1:
                good_samples += [features]
            else:
                bad_samples += [features]

    bad_samples = random.sample(bad_samples, len(good_samples))

    # convert to libsvm format
    with open('svm_train.data', 'w') as svm_fh:
    
        for data in good_samples:
            output_str = '1 ' +  '1:' + str(data[0]) + ' 2:' + str(data[1]) + ' 3:' + str(data[2]) + ' 4:' + str(data[3])
            svm_fh.write(output_str + '\n')              
        for data in bad_samples:
            output_str = '0 ' +  '1:' + str(data[0]) + ' 2:' + str(data[1]) + ' 3:' + str(data[2]) + ' 4:' + str(data[3])
            svm_fh.write(output_str + '\n')              
    

    # convert to matlab format
    with open('matlab_train.data', 'w') as fh:
    
        for data in good_samples:
            output_str = '1 ' +  str(data[0]) + ' ' + str(data[1]) + ' ' + str(data[2]) + ' ' + str(data[3])
            fh.write(output_str + '\n')              
        for data in bad_samples:
            output_str = '0 ' +  str(data[0]) + ' ' + str(data[1]) + ' ' + str(data[2]) + ' ' + str(data[3])
            fh.write(output_str + '\n')              
    


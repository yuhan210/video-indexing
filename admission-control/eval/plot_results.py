import matplotlib
import matplotlib.pyplot as plt
try:
    plt.style.use('ggplot')
except:
    pass

if __name__ == "__main__":


    # read results
     
    baseline_rate = [0.02, 0.1, 1]
    baseline_acc = [0.9, 0.94 ,1]
    baseline_accerr = [0.088, 0.086 , 0]

    greedy_rate = [0.0206959443328]
    greedy_acc = [0.88 ]
    greedy_rateerr = [ 0.00955137714151]
    greedy_accerr = [0.1413]
 
    plt.figure()
    plt.errorbar(baseline_rate, baseline_acc, yerr=baseline_accerr, color = 'b', fmt='--o', label='Uniformly')
    plt.xscale('log')
    plt.errorbar(greedy_rate, greedy_acc, xerr=greedy_rateerr, yerr=greedy_accerr, color = 'r', fmt='--x', label='Greedy')
    plt.xscale('log')
    plt.xlabel('Subsample rate')
    plt.ylabel('Averaged ranking accuracy (%)')
    plt.legend()
    plt.show()

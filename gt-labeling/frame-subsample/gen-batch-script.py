import sys
import os


for f in os.listdir(sys.argv[1]):
    
    print 'python subsample.py ' + os.path.join(sys.argv[1], f) + ' &'


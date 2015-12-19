import os
import sys


if __name__ == "__main__":
    
    for f in os.listdir(sys.argv[1]):
        old_path = os.path.join(sys.argv[1], f)
        npath = '_'.join(f.split(' '))
        new_path = os.path.join(sys.argv[1], npath)
        os.rename(old_path, new_path)

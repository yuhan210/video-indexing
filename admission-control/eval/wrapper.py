import sys
import mainscript

vgg_cache = None
fei_cache = None
msr_cache = None
rcnn_cache = None
turker_cache = None

if __name__ == "__main__":
    
    while True:
        if not vgg_cache or not fei_cache or not msr_cache or not rcnn_cache or not turker_cache:
            vgg_cache, fei_cache, msr_cache, rcnn_cache, turker_cache = mainscript.load()
        
        mainscript.run(vgg_cache, fei_cache, msr_cache, rcnn_cache, turker_cache)
        print 'Enter to re-run mainscript.run(), CTRL-C to exit'
        sys.stdin.readline()
        reload(mainscript)

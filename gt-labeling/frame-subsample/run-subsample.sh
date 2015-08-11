
#seq 1260 | xargs -i --max-procs=25 bash -c "python /home/t-yuche/gt-labeling/frame-subsample/subsample.py {}"
seq 1260 | xargs -i bash -c "python /home/t-yuche/gt-labeling/frame-subsample/subsample.py {}"

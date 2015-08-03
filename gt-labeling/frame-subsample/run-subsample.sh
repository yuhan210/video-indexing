
seq 1220 | xargs -i --max-procs=25 bash -c "python /home/t-yuche/gt-labeling/frame-subsample/subsample.py {}"

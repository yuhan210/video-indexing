
#seq 1260 | xargs -i  --max-procs=10 bash -c "python /home/t-yuche/neuraltalk/python_features/rest_vgg_classify.py {}"

seq 1260 | parallel -j 20  "python /home/t-yuche/neuraltalk/python_features/rest_vgg_classify.py" {}

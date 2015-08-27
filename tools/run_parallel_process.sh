
seq 1260 | parallel -j 25  "python /home/t-yuche/neuraltalk/python_features/rest_vgg_classify.py" {}

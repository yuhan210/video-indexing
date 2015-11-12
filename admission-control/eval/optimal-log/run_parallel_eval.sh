
seq 1260 | parallel -j 25  "python /home/t-yuche/admission-control/eval/optimal-log/create.py" {}

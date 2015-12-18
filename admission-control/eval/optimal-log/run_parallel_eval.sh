
seq 1260 | parallel -j 20  "python /home/t-yuche/admission-control/eval/optimal-log/create.py" {}

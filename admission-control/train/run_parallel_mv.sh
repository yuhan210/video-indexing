
seq 1260 | parallel -j 25  "python /home/t-yuche/admission-control/train/extract_mv_features.py" {}


seq 1260 | parallel -j 25  "python /home/t-yuche/admission-control/train/svm/code/predict.py svm_train_0.7_15283_0.5_0_100.model" {}

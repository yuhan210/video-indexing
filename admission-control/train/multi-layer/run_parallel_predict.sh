

seq 1260 | parallel -j 25  "python /home/t-yuche/admission-control/train/multi-layer/predict.py dct_2370546_0.5_eqw svm_train_0.5_10916_2_0_10.model" {}

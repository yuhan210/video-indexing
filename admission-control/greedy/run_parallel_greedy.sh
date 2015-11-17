
seq 1260 | parallel -j 25  "python /home/t-yuche/admission-control/greedy/parallel-window-greedy.py 1" {}
seq 1260 | parallel -j 25  "python /home/t-yuche/admission-control/greedy/parallel-window-greedy.py 0.9" {}
seq 1260 | parallel -j 25  "python /home/t-yuche/admission-control/greedy/parallel-window-greedy.py 0.8" {}
seq 1260 | parallel -j 25  "python /home/t-yuche/admission-control/greedy/parallel-window-greedy.py 0.7" {}
seq 1260 | parallel -j 25  "python /home/t-yuche/admission-control/greedy/parallel-window-greedy.py 0.6" {}
seq 1260 | parallel -j 25  "python /home/t-yuche/admission-control/greedy/parallel-window-greedy.py 0.5" {}
seq 1260 | parallel -j 25  "python /home/t-yuche/admission-control/greedy/parallel-window-greedy.py 0.4" {}
seq 1260 | parallel -j 25  "python /home/t-yuche/admission-control/greedy/parallel-window-greedy.py 0.3" {}
seq 1260 | parallel -j 25  "python /home/t-yuche/admission-control/greedy/parallel-window-greedy.py 0.2" {}

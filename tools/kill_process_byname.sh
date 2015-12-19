for PID in `ps ax | grep 'create.py' | awk '{print  $1;}'`; do
    kill -9 $PID;
done

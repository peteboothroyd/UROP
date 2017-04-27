#/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/home/peb42/autocity/python/autocity:$PYTHONPATH 
~/caffe/caffe/build/tools/caffe test -weights=models/_iter_10000000.caffemodel -model=models/test.prototxt -iterations=10000 --gpu $1 2>&1 | tee ./test.log

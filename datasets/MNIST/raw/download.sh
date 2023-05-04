#!/bin/bash -norc

curl -O http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz \
     -O http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz \
     -O http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz \
     -O http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
gunzip t*-ubyte.gz

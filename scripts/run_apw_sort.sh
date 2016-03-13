
source ~/.profile 

train=../../dataset/penn_wsj.dep.train
valid=../../dataset/penn_wsj.dep.valid
test=../../dataset/penn_wsj.dep.test
dataset=../../dataset/penn_wsj.dep.h5

train=/disk/scratch1/xingxing.zhang/xTreeRNN/dataset/apw/ap.dep.train
valid=/disk/scratch1/xingxing.zhang/xTreeRNN/dataset/apw/ap.dep.valid
test=/disk/scratch1/xingxing.zhang/xTreeRNN/dataset/apw/ap.dep.test
dataset=/disk/scratch1/xingxing.zhang/xTreeRNN/dataset/apw/ap.dep.h5
th deptree2hdf5.lua --train $train --valid $valid --test $test --dataset $dataset --freq 20 --keepFreq --maxLen 100 --ignoreCase --sort 0 --batchSize 64


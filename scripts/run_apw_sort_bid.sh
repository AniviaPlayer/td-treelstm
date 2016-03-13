
source ~/.profile 

train=/disk/scratch1/xingxing.zhang/xTreeRNN/dataset/apw/ap.dep.train
valid=/disk/scratch1/xingxing.zhang/xTreeRNN/dataset/apw/ap.dep.valid
test=/disk/scratch1/xingxing.zhang/xTreeRNN/dataset/apw/ap.dep.test
dataset=/disk/scratch1/xingxing.zhang/xTreeRNN/dataset/apw/ap.dep.bid.h5
th deptree2hdf5.lua --train $train --valid $valid --test $test --dataset $dataset --freq 20 --keepFreq --maxLen 100 --ignoreCase --sort 0 --batchSize 64 --bidirectional

# due to a strange bug; we must sort the h5 file with an individual program
th sort_large_hdf5_bid.lua --dataset $dataset --sort -1 --batchSize 64 --bidirectional


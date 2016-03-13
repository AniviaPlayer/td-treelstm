
source ~/.profile 

datadir=/afs/inf.ed.ac.uk/group/project/img2txt/deptree_rnnlm/data/msr_sent_compl
train=$datadir/msr.dep.100.train
valid=$datadir/msr.dep.100.valid
test=$datadir/msr.dep.100.test
dataset=/disk/scratch/XingxingZhang/treelstm/dataset/msr/msr.dep.100.bid.h5

train=../../dataset/penn_wsj.dep.train
valid=../../dataset/penn_wsj.dep.valid
test=../../dataset/penn_wsj.dep.test
dataset=../../dataset/penn_wsj.dep.bid.h5
th deptree2hdf5.lua --train $train --valid $valid --test $test --dataset $dataset --freq 5 --keepFreq --maxLen 150 --ignoreCase --sort -1 --batchSize 64 --bidirectional



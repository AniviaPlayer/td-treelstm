
ID=`./gpu_lock.py --id-to-hog 0`
echo $ID
if [ $ID -eq -1 ]; then
    echo "no gpu is free"
    exit
fi
./gpu_lock.py

curdir=`pwd`
codedir=/afs/inf.ed.ac.uk/group/project/img2txt/xTreeLSTM/xtreelstm/td-treelstm-release
dataset=/disk/scratch/XingxingZhang/treelstm/dataset/msr/msr.dep.100.bid.sort20.h5
lr=1.0
label=.ldtree.400
model=model_$lr$label.t7
log=$lr$label.txt

cd $codedir

CUDA_VISIBLE_DEVICES=$ID th $codedir/main_bid.lua \
    --model BiTreeLSTMNCE \
    --dataset $dataset \
    --useGPU \
    --nin 200 \
    --nhid 400 \
    --nlayers 1 \
    --lr $lr \
    --batchSize 64 \
    --validBatchSize 16 \
    --maxEpoch 50 \
    --save $curdir/$model \
    --gradClip 5 \
    --optimMethod SGD \
    --patience 1 \
    --nneg 20 \
    --power 0.75 \
    --lnZ 9 \
    --learnZ \
    --savePerEpoch \
    --saveBeforeLrDiv \
    --seqLen 101 \
    --nlclayers 1 \
    --nlchid 400 \
    | tee $curdir/$log

./gpu_lock.py --free $ID
./gpu_lock.py

testfile=/disk/scratch/XingxingZhang/treelstm/dataset/msr/msr.dep.100.bid.question.h5
outfile=out.txt
rlog=rerank.$log

time th $codedir/rerank.lua --modelPath $curdir/$model \
    --testFile $testfile \
    --outFile $curdir/$outfile | tee $curdir/$rlog

cd $curdir


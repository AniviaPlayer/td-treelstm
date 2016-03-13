
ID=`./gpu_lock.py --id-to-hog 1`
echo $ID
if [ $ID -eq -1 ]; then
    echo "no gpu is free"
    exit
fi
./gpu_lock.py

curdir=`pwd`
codedir=/afs/inf.ed.ac.uk/group/project/img2txt/xTreeLSTM/xtreelstm/td-treelstm-release
dataset=/disk/scratch/XingxingZhang/treelstm/dataset/depparse/dataset/penn_wsj.conllx.sort.h5
wembed=/disk/scratch/XingxingZhang/treelstm/dataset/res/glove/glove.6B.100d.t7
lr=1.0
label=.w200
model=model_$lr$label.t7
log=$lr$label.txt

cd $codedir

CUDA_VISIBLE_DEVICES=$ID th $codedir/main_nce.lua \
    --model TreeLSTM \
    --dataset $dataset \
    --wordEmbedding $wembed \
    --useGPU \
    --nin 100 \
    --nhid 200 \
    --nlayers 2 \
    --lr $lr \
    --batchSize 64 \
    --maxEpoch 50 \
    --save $curdir/$model \
    --gradClip 5 \
    --optimMethod SGD \
    --patience 1 \
    --dropout 0.2 \
    | tee $curdir/$log

cd $curdir

./gpu_lock.py --free $ID
./gpu_lock.py


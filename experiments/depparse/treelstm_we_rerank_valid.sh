
ID=`./gpu_lock.py --id-to-hog 1`
echo $ID
if [ $ID -eq -1 ]; then
    echo "no gpu is free"
    exit
fi
./gpu_lock.py

curdir=`pwd`
codedir=/afs/inf.ed.ac.uk/group/project/img2txt/xTreeLSTM/xtreelstm/td-treelstm-release

vocab=/disk/scratch/XingxingZhang/treelstm/dataset/depparse/dataset/penn_wsj.conllx.sort.vocab.t7
basefile=/disk/scratch/XingxingZhang/treelstm/experiments/ptb_depparse/rerank_data/dev-20-best-mst2ndorder.conll.g
model=/disk/scratch/XingxingZhang/treelstm/experiments/ptb_depparse/2layer_w_wo_we/model_1.0.w200.t7
label=.w200.2l
log=log$label.txt
scorefile=out$label.txt

goldfile=/disk/scratch/XingxingZhang/treelstm/experiments/ptb_depparse/rerank_data/valid
basescorefile=/disk/scratch/XingxingZhang/treelstm/experiments/ptb_depparse/rerank_data/dev-20-best-mst2ndorder.conll.mstscores

cd $codedir

CUDA_VISIBLE_DEVICES=$ID th $codedir/dep_rerank.lua \
    --vocab $vocab \
    --baseFile $basefile \
    --modelPath $model \
    --useGPU \
    --batchSize 64 \
    --scoreFile $curdir/$scorefile \
    --baseScoreFile $basescorefile \
    --goldFile $goldfile \
    --k 20 \
    --searchk \
    --standard stanford \
    | tee $curdir/$log

cd $curdir

./gpu_lock.py --free $ID
./gpu_lock.py


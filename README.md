Top-down Tree Long Short-Term Memory Networks
===============================================


A [Torch](https://github.com/torch) implementation of the Top-down TreeLSTM described in the following paper. 

### [Top-down Tree Long Short-Term Memory Networks](http://arxiv.org/abs/1511.00060)
Xingxing Zhang, Liang Lu and Mirella Lapata. In *Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies* (NAACL 2016).

### Implemented Models 
* TreeLSTM
* TreeLSTM-NCE
* LdTreeLSTM
* LdTreeLSTM-NCE

TreeLSTM and LdTreeLSTM (check the details in the paper above) are trained with Negative Log-likelihood (NLL); while TreeLSTM-NCE and LdTreeLSTM-NCE are trained with [Noise Contrastive Estimation](https://www.cs.helsinki.fi/u/ahyvarin/papers/Gutmann12JMLR.pdf) (NCE) (see [this paper](https://www.cs.helsinki.fi/u/ahyvarin/papers/Gutmann12JMLR.pdf) and also [this paper](https://www.cs.toronto.edu/~amnih/papers/ncelm.pdf) for details).

Note that in experiments, the normalization term Z of NCE is learned automatically. The implemented NCE module also support keeping Z fixed.

# Requirements
* a Nvidia GPU
* [CUDA 6.5.19](http://www.nvidia.com/object/cuda_home_new.html) (higher version should be fine)
* [Torch](https://github.com/torch)
* [torch-hdf5](https://github.com/deepmind/torch-hdf5)

Torch can be installed with the instructions [here](http://torch.ch/docs/getting-started.html). 
You also need to install some torch components.
```
luarocks install nn
luarocks install nngraph
luarocks install cutorch
luarocks install cunn
```
You may find [this document](https://github.com/deepmind/torch-hdf5/blob/master/doc/usage.md) useful when installing torch-hdf5 (DON'T use luarocks).

Please also note that to run the code, you need to use an old version of Torch with the instructions [here](OLD_VERSION.md).


# Language Modeling (MSR Sentence Completion)

## Preprocessing
First, parse the dataset into dependency trees using [Stanford CoreNLP toolkit](http://stanfordnlp.github.io/CoreNLP/).
It should looks like this
```
SILAP10.TXT#0	det(Etext-4, The-1) nn(Etext-4, Project-2) nn(Etext-4, Gutenberg-3) root(ROOT-0, Etext-4) prep(Etext-4, of-5) det(Rise-7, The-6) pobj(of-5, Rise-7) prep(Rise-7, of-8) nn(Lapham-10, Silas-9) pobj(of-8, Lapham-10) prep(Etext-4, by-11) nn(Howells-14, William-12) nn(Howells-14, Dean-13) pobj(by-11, Howells-14) det(RISE-16, THE-15) dep(Etext-4, RISE-16) prep(RISE-16, OF-17) nn(LAPHAM-19, SILAS-18) pobj(OF-17, LAPHAM-19) prep(RISE-16, by-20) nn(Howells-23, William-21) nn(Howells-23, Dean-22) pobj(by-20, Howells-23) npadvmod(Howells-23, I-24) punct(Etext-4, .-25)
SILAP10.TXT#1	advmod(went-4, WHEN-1) nn(Hubbard-3, Bartley-2) nsubj(went-4, Hubbard-3) advcl(received-40, went-4) aux(interview-6, to-5) xcomp(went-4, interview-6) nn(Lapham-8, Silas-7) dobj(interview-6, Lapham-8) prep(interview-6, for-9) det(Men-13, the-10) punct(Men-13, ``-11) amod(Men-13, Solid-12) pobj(for-9, Men-13) prep(Men-13, of-14) pobj(of-14, Boston-15) punct(Men-13, ''-16) dep(Men-13, series-17) punct(series-17, ,-18) dobj(undertook-21, which-19) nsubj(undertook-21, he-20) rcmod(series-17, undertook-21) aux(finish-23, to-22) xcomp(undertook-21, finish-23) prt(finish-23, up-24) prep(finish-23, in-25) det(Events-27, The-26) pobj(in-25, Events-27) punct(received-40, ,-28) mark(replaced-31, after-29) nsubj(replaced-31, he-30) advcl(received-40, replaced-31) poss(projector-34, their-32) amod(projector-34, original-33) dobj(replaced-31, projector-34) prep(replaced-31, on-35) det(newspaper-37, that-36) pobj(on-35, newspaper-37) punct(received-40, ,-38) nsubj(received-40, Lapham-39) root(ROOT-0, received-40) dobj(received-40, him-41) prep(received-40, in-42) poss(office-45, his-43) amod(office-45, private-44) pobj(in-42, office-45) prep(received-40, by-46) amod(appointment-48, previous-47) pobj(by-46, appointment-48) punct(received-40, .-49)
...
...
...
```
Each line is a sentence (format: label \t dependency tuples), where *SILAP10.TXT#0* is the label for the sentence (it can be any string and it doesn't matter).

Dataset after the preprocessing above can be downloaded [here](https://drive.google.com/file/d/0B6-YKFW-MnbONGpFblhJRUFtRjQ/view?usp=sharing).

Then, convert the dependency tree dataset into HDF5 format and sort the dataset to make sure sentences in each batch have similar length. Sorting the dataset is for faster training, which is a commonly used strategy for training RNN or Sequence based models.

### Create Dataset for TreeLSTM
```
cd scripts
./run_msr.sh
./run_msr_sort.sh
./run_msr_test.sh
```
Note the program will crash when running *./run_msr_sort.sh*. You can ignore the crash or you should use *--sort 0* switch instead of *--sort 20*.

### Create Dataset for LdTreeLSTM
```
cd scripts
./run_msr_bid.sh
./run_msr_sort_bid.sh
./run_msr_test_bid.sh
```
Alternately, you can contact the first author to request the dataset after preprocessing.

## Training and Evaluation
Basically it is just one command.
```
cd experiments/msr
# to run TreeLSTM with hidden size 400
./treelstm_h400.sh
# to run LdTreeLSTM with hidden size 400
./ldtreelstm_h400.sh
```
But don't forget to specify where is your code, your dataset and whatever by modifying treelstm_h400.sh or ldtreelstm_h400.sh.
```
# where is your code? (you should use absolute path)
codedir=/afs/inf.ed.ac.uk/group/project/xTreeLSTM/xtreelstm/td-treelstm-release
# where is your dataset (you should use absolute path)
dataset=/disk/scratch/XingxingZhang/treelstm/dataset/msr/msr.dep.100.bid.sort20.h5
# label for this model
label=.ldtree.400

# where is your testset (you should use absolute path); this will only be used in evaluation
testfile=/disk/scratch/XingxingZhang/treelstm/dataset/msr/msr.dep.100.bid.question.h5
```
# Dependency Parsing Reranking

## Preprocessing

### For TreeLSTM
```
cd scripts
./run_conllx_sort.sh
```

### For LdTreeLSTM
```
cd scripts
./run_conllx_sort_bid.sh
```

## Train and Evaluate Dependency Reranking Models
Training TreeLSTMs and LdTreeLSTMs are quit similar. 
The following is about training a TreeLSTM.
```
cd experiments/depparse
./treelstm_we_train.sh

```
Then, you will get a trained TreeLSTM. We can use this TreeLSTM
to rerank the *K* dependencies produced by the second order [MSTParser](http://www.seas.upenn.edu/~strctlrn/MSTParser/MSTParser.html).

The following script will use the trained dependency model to rerank the top 20 dependencies from MSRParser on the validation set. The script will try different *K* and choose the one gives best UAS.
```
./treelstm_we_rerank_valid.sh
```
Given the *K* we've got from the validation set, we can get the reranking performance on test set by using the following script.
```
./treelstm_we_rerank_test.sh
```

# Dependency Tree Generation

### How will we generate dependency trees? (details see Section 3.4 of the paper)
* Run the Language Modeling experiment or the dependency parsing experiment to get a trained TreeLSTM or LdTreeLSTM
* Generate training data for the four classifiers (Add-Left, Add-Right, Add-Nx-Left, Add-Nx-Right)
* Train Add-Left, Add-Right, Add-Nx-Left and Add-Nx-Right
* Generate dependency trees with a trained TreeLSTM (or LdTreeLSTM) and the four classifiers

### Generate Training data
Go to *sampler.lua* and run the following code
```
-- model_1.0.w200.t7 is the trained TreeLSTM
-- penn_wsj.conllx.sort.h5 is the dataset for the trained TreeLSTM
-- eot.penn_wsj.conllx.sort.h5 is the output dataset for the four classifiers
generateDataset('/disk/scratch/XingxingZhang/treelstm/experiments/ptb_depparse/2layer_w_wo_we/model_1.0.w200.t7', 
    '/disk/scratch/XingxingZhang/treelstm/dataset/depparse/dataset/penn_wsj.conllx.sort.h5', 
    '/disk/scratch/XingxingZhang/treelstm/dataset/depparse/eot.penn_wsj.conllx.sort.h5')

```

### Train the Four Classifiers
Use *train_mlp.lua*
```
$ th train_mlp.lua -h
Usage: /afs/inf.ed.ac.uk/user/s12/s1270921/torch/install/lib/luarocks/rocks/trepl/scm-1/bin/th [options] 
====== MLP v 1.0 ======

  --seed        random seed [123]
  --useGPU      use gpu [false]
  --snhids      string hidden sizes for each layer [400,300,300,2]
  --activ       options: tanh, relu [tanh]
  --dropout     dropout rate (dropping) [0]
  --maxEpoch    max number of epochs [10]
  --dataset     dataset [/disk/scratch/XingxingZhang/treelstm/dataset/depparse/eot.penn_wsj.conllx.sort.h5]
  --ftype        [|x|oe|]
  --ytype        [1]
  --batchSize    [256]
  --lr           [0.01]
  --optimMethod options: SGD, AdaGrad [AdaGrad]
  --save        save path [model.t7]

```
Note *--ytype* 1, 2, 3, 4 corresponds to the four classifiers. Here is a sample script:
```
ID=`./gpu_lock.py --id-to-hog 2`
echo $ID
if [ $ID -eq -1 ]; then
    echo "no gpu is free"
    exit
fi
./gpu_lock.py

curdir=`pwd`
codedir=/afs/inf.ed.ac.uk/group/project/xTreeLSTM/xtreelstm/MLP_test
lr=0.01
label=yt1.x.oe
model=model.$label.t7
log=log.$label.txt
echo $curdir
echo $codedir

cd $codedir
CUDA_VISIBLE_DEVICES=$ID th train_mlp.lua --useGPU \
    --activ relu --dropout 0.5 --lr $lr --maxEpoch 10 \
    --snhids "400,300,300,2" --ftype "|x|oe|" --ytype 1 \
    --save $curdir/$model | tee $curdir/$log

cd $curdir

./gpu_lock.py --free $ID
./gpu_lock.py

```
### Generation by Sampling
Go to *sampler.lua* and run the following code. The code will output dependency trees in LaTeX format.
```
-- model_1.0.w200.t7: trained TreeLSTM
-- model.yt%d.x.oe.t7: trained classifiers, note that model.yt1.x.oe.t7, model.yt2.x.oe.t7, model.yt3.x.oe.t7 and model.yt4.x.oe.t7 must all exist
sampleTrees('/disk/scratch/XingxingZhang/treelstm/experiments/ptb_depparse/2layer_w_wo_we/model_1.0.w200.t7',
    '/disk/scratch/XingxingZhang/treelstm/experiments/sampling/eot_classify/model.yt%d.x.oe.t7',
    's100.txt', -- output dependency trees
    1,     -- rand seed
    100)   -- number of tree samples
```

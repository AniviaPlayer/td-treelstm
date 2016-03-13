
### Using Old Version Torch
Torch is under activate development. Unfortunately, some APIs in later versions are not compatible with these in the earlier versions, which is terrible!!! In order to run the code,
you need to go back to the version around 2015-07-22 by running the following commands:
```
# create a directory for torch source code
mkdir src
cd src
git clone https://github.com/torch/torch7.git
git clone https://github.com/torch/nn.git
git clone https://github.com/torch/cutorch.git
git clone https://github.com/torch/nngraph.git
git clone https://github.com/torch/cunn.git

cd torch7/
git checkout 80a545e
luarocks make rocks/torch-scm-1.rockspec
cd ..

cd nn
git checkout c503fb8
luarocks make rocks/nn-scm-1.rockspec
cd ..

cd cutorch
git checkout 2eddb66
luarocks make rocks/cutorch-scm-1.rockspec
cd ..

cd cunn
git checkout 4f66456
luarocks make rocks/cunn-scm-1.rockspec
cd ..

cd nngraph
git checkout 1c43c98
luarocks make nngraph-scm-1.rockspec
cd ..
```

[README.md](README.md)
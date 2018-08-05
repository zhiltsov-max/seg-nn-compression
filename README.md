# Semantic segmentation networks compression

This repository contains an implementation details of the semantic segmentation task solver, 
based on the Torch framework. 

# Repo structure

- models - Contains some experimental models
- datasets - Contains dataset adapters
    - datasets/$datasetName/list - Files with images lists like 'train', 'val' and 'test'
    - datasets/$datasetName/<data folders as specified in lists>
- main.lua - Main executable script
- train.lua - Solver and optimizer
- test.lua - Inference script


# Installation

## Dependencies

- Torch 7 [http://torch.ch/docs/getting-started.html]. TL;DR:
   ``` bash
   git clone https://github.com/torch/distro.git ~/torch --recursive
   cd ~/torch; bash install-deps;
   ./install.sh
   ```
- Torch packages: 
    - optnet
    - image
    - cutorch
    - cudnn
    ``` bash
    luarocks install <package name>
    ```

- NVIDIA CUDA (tested on 8.0)
- NVIDIA CuDNN (tested on 5.0)


## Project

``` bash
git clone https://github.com/zhiltsov-max/seg-nn-compression.git
```

## Datasets

### CamVid

Modified version of CamVid dataset with 12 semantic classes presented in [SegNet paper](http://arxiv.org/abs/1511.00561).

```
git clone https://github.com/alexgkendall/SegNet-Tutorial

mkdir -p seg-nn-compression/data/camvid12/list
ln -s SegNet-Tutorial seg-nn-compression/data/camvid/SegNet
cp SegNet-Tutorial/CamVid/*.txt seg-nn-compression/data/camvid12/list
```

# Run

Check `exec.sh` for example and `main.lua` for execution options.

``` bash
th main.lua <options>
```

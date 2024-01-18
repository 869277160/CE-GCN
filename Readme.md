<!--
 * @Author: your name
 * @Date: 2022-04-03 17:32:34
 * @LastEditTime: 2024-01-18 20:47:00
 * @LastEditors: wangding wangding19@mails.ucas.ac.cn
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: /CEGCN/Readme.md
-->
# CE-GCN

![](https://img.shields.io/badge/python-3.9.7-green)

This repo provides a reference implementation of **CE-GCN** as described in the paper:
> [Cascade-Enhanced Graph Convolutional Network for Information Diffusion Prediction](https://doi.org/10.1007978-3-031-00123-9_50)


## Basic Usage

## DATASET
You can find the dataset in the "data" folder, which contains all three datasets (Twitter, Douban, and Meme).

## Environmental Settings
Our experiments are conducted on CentOS 20.04, a single NVIDIA V100 GPU. CCGL is implemented by `Python 3.7`, `Torch 1.0.9`.

Create a virtual environment and install GPU-support packages via [Anaconda](https://www.anaconda.com/):
```shell
# create virtual environment
conda create --name=CEGCN python=3.9

# activate virtual environment
conda activate CEGCN

# install other related dependencies
conda install wandb

conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch

conda install pyg -c pyg -c conda-forge

conda install scikit-learn-intelex
```


## Usage

You can run our model with the following commands:
```shell
CUDA_VISIBLE_DEVICES=1 python run.py --data="twitter"
CUDA_VISIBLE_DEVICES=1 python run.py --data="douban"
CUDA_VISIBLE_DEVICES=1 python run.py --data="memetracker"

CUDA_VISIBLE_DEVICES=1 nohup python run.py --data="twitter" &
CUDA_VISIBLE_DEVICES=0 nohup python run.py --data="douban" &
CUDA_VISIBLE_DEVICES=1 nohup python run.py --data="memetracker
```

### running with different graphs

```shell
CUDA_VISIBLE_DEVICES=1 nohup  python run.py --data="twitter" --notes="item" &
CUDA_VISIBLE_DEVICES=0 nohup  python run.py --data="twitter" --notes="social" &
CUDA_VISIBLE_DEVICES=0 nohup  python run.py --data="twitter" --notes="diffusion" &
CUDA_VISIBLE_DEVICES=1 nohup  python run.py --data="twitter" --notes="social+item" &
CUDA_VISIBLE_DEVICES=1 nohup  python run.py --data="twitter" --notes="diffusion+item" &
CUDA_VISIBLE_DEVICES=1 nohup  python run.py --data="twitter" --notes="social+diffusion" &

CUDA_VISIBLE_DEVICES=1 nohup  python run.py --data="douban" --notes="item" &
CUDA_VISIBLE_DEVICES=0 nohup  python run.py --data="douban" --notes="social" &
CUDA_VISIBLE_DEVICES=1 nohup  python run.py --data="douban" --notes="diffusion" &
CUDA_VISIBLE_DEVICES=0 nohup  python run.py --data="douban" --notes="social+item" &
CUDA_VISIBLE_DEVICES=1 nohup  python run.py --data="douban" --notes="diffusion+item" &
CUDA_VISIBLE_DEVICES=1 nohup  python run.py --data="douban" --notes="social+diffusion" &
```


## Cite

If you find our paper & code are useful for your research, please consider citing us 😘:

```bibtex
@inproceedings{DBLP:conf/dasfaa/WangWYBZZH22,
  author       = {Ding Wang and
                  Lingwei Wei and
                  Chunyuan Yuan and
                  Yinan Bao and
                  Wei Zhou and
                  Xian Zhu and
                  Songlin Hu},
  editor       = {Arnab Bhattacharya and
                  Janice Lee and
                  Mong Li and
                  Divyakant Agrawal and
                  P. Krishna Reddy and
                  Mukesh K. Mohania and
                  Anirban Mondal and
                  Vikram Goyal and
                  Rage Uday Kiran},
  title        = {Cascade-Enhanced Graph Convolutional Network for Information Diffusion
                  Prediction},
  booktitle    = {Database Systems for Advanced Applications - 27th International Conference,
                  {DASFAA} 2022, Virtual Event, April 11-14, 2022, Proceedings, Part
                  {I}},
  series       = {Lecture Notes in Computer Science},
  volume       = {13245},
  pages        = {615--631},
  publisher    = {Springer},
  year         = {2022},
  url          = {https://doi.org/10.1007/978-3-031-00123-9\_50},
  doi          = {10.1007/978-3-031-00123-9\_50},
  timestamp    = {Fri, 29 Apr 2022 14:50:40 +0200},
  biburl       = {https://dblp.org/rec/conf/dasfaa/WangWYBZZH22.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

## Contact

For any questions please open an issue or drop an email to: `wangding@iie.ac.cn`






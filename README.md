# RankSpace-Models
This repository contains source code for NAACL2022 main conference [Dynamic Programming in Rank Space: Scaling Structured Inference with Low-Rank HMMs and PCFGs](https://faculty.sist.shanghaitech.edu.cn/faculty/tukw/naacl22rank.pdf)

* Code for scaling low-rank structured inference with HMMs in rank-hmms.
* Code for scaling low-rank structured inference with PCFGs in rank-pcfgs, which has been merged into [TN-PCFG](https://github.com/sustcsonglin/TN-PCFG). More details can be seen in it.

## Dependencies
```
pip install -r requirement.txt
```

## HMMs
### Prepare Data
If your running environment supports internet connection, you can just run the train. Then corresponding dataset will be downloaded automatically.

Otherwise, download dataset [here](https://drive.google.com/file/d/1f0nKHmXrBIs-LkH4XikYoM8xKM49FhA5/view?usp=sharing). And then change the name of directory `data` to `.data`.

### How to run
```
python train.py --conf path/to/config.yaml --d cuda device number --version "any name you like"
e.g.
python train.py --conf ./config/projrank_m32768_r4096.yaml -d 0 --version rank-hmms
```

## PCFGs
We have merged the rank-pcfgs into [TN-PCFG](https://github.com/sustcsonglin/TN-PCFG). Please see more details there.

## Acknowledge
The code is based on [low-rank-models](https://github.com/justinchiu/low-rank-models), [hmmlm](https://github.com/harvardnlp/hmm-lm). And most baselines for HMMs can be found there.

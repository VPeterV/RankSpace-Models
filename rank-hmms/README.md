# HMMs
## Prepare Data
If your running environment supports internet connection, you can just run the train. Then corresponding dataset will be downloaded automatically.

Otherwise, download dataset [here](https://drive.google.com/file/d/1f0nKHmXrBIs-LkH4XikYoM8xKM49FhA5/view?usp=sharing). And then change the name of directory `data` to `.data`.

## How to run
```
python train.py --conf path/to/config.yaml --d cuda device number --version "any name you like"
e.g.
python train.py --conf ./config/projrank_m32768_r4096.yaml -d 0 --version rank-hmms
```
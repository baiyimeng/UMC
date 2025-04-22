# UMC
[![arXiv](https://img.shields.io/badge/arXiv-2504.14243-red.svg)](https://arxiv.org/abs/2504.14243)

This is the pytorch implementation of our paper at SIGIR 2025:
> [Unconstrained Monotonic Calibration of Predictions in Deep Ranking Systems](https://arxiv.org/abs/2504.14243)
> 
> Yimeng Bai, Shunyu Zhang, Yang Zhang, Hu Liu, Wentian Bao, Enyun Yu, Fuli Feng, Wenwu Ou.

## Usage
### Data
The experimental datasets are available for download via the links provided in the files located at `/data/dataset/download.txt`.
### Training & Evaluation
First, train the backbone ranking model [DeepFM](https://arxiv.org/abs/1703.04247) on the **Avazu** or **AliCCP** dataset, implemented based on [DeepCTR-Torch](https://github.com/shenweichen/DeepCTR-Torch).
```
python pretrain.py
```
Then, train the calibrator model. The proposed **UMC**, together with the deep-learning based baseline methods **FAC**, **SBCR**, and **DESC**, is implemented in the `train_neu_avazu.py` and `train_neu_ali.py` script. In contrast, the remaining baseline methods are implemented in `train_sta_avazu.py` and `train_sta_ali.py`
```
python train_neu_avazu.py
python train_sta_avazu.py
python train_neu_ali.py
python train_sta_ali.py
```
## Citation
```
@inproceedings{UMC,
author = {Bai, Yimeng and Zhang, Shunyu and Zhang, Yang and Liu, Hu and Bao, Wentian and Yu, Enyun and Feng, Fuli and Ou, Wenwu},
title = {Unconstrained Monotonic Calibration of Predictions in Deep Ranking Systems},
year = {2025},
isbn = {9798400715921},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3726302.3730105},
doi = {10.1145/3726302.3730105},
booktitle = {Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval},
numpages = {11},
keywords = {calibrator modeling, unconstrained monotonic neural network,
ranking system},
location = {Padua, Italy},
series = {SIGIR '25}
}
```


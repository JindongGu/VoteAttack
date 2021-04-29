# VoteAttack on Capsule Network

## This is a PyTorch implementation of the paper "Effective and Efficient Vote Attack on Capsule Networks" [ICLR 2021]

### Evaluating the robustness of the model

Evaluating the pre-trained model with different attack methods
```
python main.py  --evaluate --dataset cifar10  --eps 0.031  --model capsnet  --attack vote_attack_FGSM
```
dataset: cifar10 (with eps=0.031), svhn (with eps=0.047) <br />
model: capsnet, resnet <br />
attack: FGSM, PGD, vote_attack_FGSM, vote_attack_PGD

### Training the model
Training the capsNet on cifar10 dataset
```
python main.py  --dataset cifar10  --model capsnet 
```

Please consider citing our paper
```
@inproceedings{gu2021effective,
title={Effective and Efficient Vote Attack on Capsule Networks},
author={Jindong Gu and Baoyuan Wu and Volker Tresp},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=33rtZ4Sjwjn}
}
```

Contact: jindong.gu@outlook.com



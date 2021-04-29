# VoteAttack

## This is a PyTorch implementation of the paper "Effective and Efficient Vote Attack on Capsule Networks" [ICLR 2021]

### Evaluating the robustness of the model

Evaluating the pre-trained model with different attack methods
```
python main.py  --evaluate --dataset cifar10  --eps 0.031  --model capsnet  --attack vote_attack_FGSM
```
model: capsnet, resnet

dataset: cifar10, svhn

attack: FGSM, PGD, vote_attack_FGSM, vote_attack_PGD

### Training the model
Training the capsNet on cifar10 dataset
```
python main.py  --dataset cifar10  --model capsnet 
```

Contact: jindong.gu@outlook.com



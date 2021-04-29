#§ encoding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import argparse
from tqdm import tqdm
from load_data import load_data

from models import resnet, capsnet
from modules import train_epoch, test_epoch, load_save_model, adjust_learning_rate

import foolbox
from foolbox.criteria import Misclassification, TargetedMisclassification


# Training settings
parser = argparse.ArgumentParser(description='Robust CapsNets')
parser.add_argument('--dataset', type=str, default='cifar10', help='the name of dataset')
parser.add_argument('--data_folder', type=str, default='./data/', metavar='DF', help='where to store the datasets')
parser.add_argument('--batch_size', type=int, default=256, metavar='N', help='input batch size for training')
parser.add_argument('--test_batch_size', type=int, default=256, metavar='N', help='input batch size for testing')

parser.add_argument('--epochs', type=int, default=80, metavar='N', help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--evaluate', action='store_true', default=False)

parser.add_argument('--model', type=str, default='capsnet', help='the name of model type: resnet, capsnet')
parser.add_argument('--num_class', type=int, default=10, metavar='N', help='the number of output classes')
parser.add_argument('--routing', type=str, default='DR', help='the routing name')

parser.add_argument('--attack', type=str, default='PGD', help='PGD, CW, FGSM, Deepfool')
parser.add_argument('--eps', type=float, default=0.031, help='the perturbation threshold')
args = parser.parse_args()


def main():

    torch.backends.cudnn.benchmark=True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = load_data(args)

    if args.model == 'resnet':
        args.routing = 'No'
        model = resnet.ResNet18()
        loss_func = nn.CrossEntropyLoss().to(device)
    elif args.model == 'capsnet':
        args.routing = 'DR'
        model = capsnet.Capsnet(args.routing)
        loss_func = nn.NLLLoss().to(device)

    model = model.to(device)
            
    opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    if args.evaluate:
        model = load_save_model(model, args, mode='load')

        test_accu, test_loss = test_epoch(test_loader, model, device=device, use_tqdm=True)
        print('Standard Test Accu', test_accu)

        if 'vote_attack' in args.attack:
            # VoteAttack: Attack Votes directly (e.g., vote_attack_PGD)
            # We construct a model where the votes are averaged to form outputs
            # The adversarial examples are created on the constructed model and tested on the original CapsNet
            model_source = capsnet.Capsnet('Vote_Attack')
            model_source.load_state_dict(model.state_dict())
        else:
            # CapsAttack: Attack output capsules (e.g., PGD)
            # The adversarial examples are created on the model itself. 
            model_source = model

        model_source.eval()
        fmodel = foolbox.models.PyTorchModel(model_source, bounds=(0., 1.), device=device)

        if 'FGSM' in args.attack: attack = foolbox.attacks.LinfFastGradientAttack()
        elif 'BIM' in args.attack: attack = foolbox.attacks.LinfBasicIterativeAttack(abs_stepsize=0.01)
        elif 'PGD' in args.attack: attack = foolbox.attacks.LinfProjectedGradientDescentAttack(abs_stepsize=0.01)

        robust_accu = 0
        total_n = 0
        for X,y in tqdm(test_loader):
            X, y = X.to(device), y.to(device)
            
            criterion = Misclassification(y)
            delta, x_adv, success = attack(fmodel, X, criterion, epsilons=args.eps)
            yp = model(x_adv.detach()).detach()

            robust_accu += (yp.max(dim=1)[1] == y).sum().item()        
            total_n += yp.shape[0]

        print('Robust Accu', robust_accu / total_n)
        return

    # Start Traning
    print('Start Standard Training ---------------------')
    for ep in range(1, args.epochs):
 
        adjust_learning_rate(ep, opt, args)

        train_accu, train_loss = train_epoch(train_loader, model, opt, device=device, use_tqdm=True, loss_func=loss_func)
        
        test_accu, test_loss = test_epoch(test_loader, model, device=device, use_tqdm=False, loss_func=loss_func)

        print('\nEpoch %d:'%ep, 'Train Accu: %.4f'%train_accu, 'Train Loss: %.4f'%train_loss, 'Test Accu: %.4f'%test_accu, 'Test Loss: %.4f'%test_loss)

    load_save_model(model, args, mode='save')

    
if __name__ == '__main__':
    main()

    

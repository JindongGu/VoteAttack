#§ encoding: utf-8
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_data(args):
    norm_mean = 0
    norm_var = 1
    if args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((norm_mean, norm_mean, norm_mean), (norm_var, norm_var, norm_var)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((norm_mean,norm_mean,norm_mean), (norm_var, norm_var, norm_var)),
        ])
        
        train_dataset = datasets.CIFAR10(args.data_folder, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(args.data_folder, train=False, download=True, transform=transform_test)
        
    elif args.dataset == 'svhn':
        transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((norm_mean,norm_mean,norm_mean), (norm_var, norm_var, norm_var)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((norm_mean,norm_mean,norm_mean), (norm_var, norm_var, norm_var)),
        ])
        
        train_dataset = datasets.SVHN(args.data_folder, split='train', download=True, transform=transform_train)
        test_dataset = datasets.SVHN(args.data_folder, split='test', download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size = args.test_batch_size, shuffle=True, num_workers=0, pin_memory=True)

    print('the number of total training examples: ', len(train_loader.dataset))
    return train_loader, test_loader






import torch
import random

import torchvision
import torchvision.transforms as transforms
from utils import Cutout


class CifarDataset():
    def __init__(self, ROOTDIR,  bCutOut=True, cutout_length=16, numberworks=1, dataset_name='cifar10'):
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.datasetNumberClass = 10
        self.numberwork = numberworks
        self.bCutOut = bCutOut

        print('==> Preparing data..')
        CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
        CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            #
        ])
        if self.bCutOut:
            transform_train.transforms.append(Cutout(cutout_length))

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

        #划分数据
        if dataset_name=='cifar100':
            self.trainset = torchvision.datasets.CIFAR100(root=ROOTDIR, train=True, download=True, transform=transform_train)
            self.testset = torchvision.datasets.CIFAR100(root=ROOTDIR, train=False, download=True, transform=transform_test)
        else:
            self.trainset = torchvision.datasets.CIFAR10(root=ROOTDIR, train=True, download=True, transform=transform_train)
            self.testset = torchvision.datasets.CIFAR10(root=ROOTDIR, train=False, download=True, transform=transform_test)


    # dataloader
    def getDataLoader(self, train_batchsize=None, val_batchsize=None, test_batchsize=None):
        indices = list(range(len(self.trainset)))
        random.shuffle(indices)
        split = int(0.7 * len(self.trainset))
        trainSampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
        testSampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])

        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=train_batchsize, num_workers=self.numberwork, sampler=trainSampler)
        self.validloader = torch.utils.data.DataLoader(self.trainset, batch_size=val_batchsize, num_workers=self.numberwork, sampler=testSampler)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=test_batchsize, shuffle=False, num_workers=self.numberwork)

        return self.trainloader, self.validloader, self.testloader

    # dataloader
    def getFixDataLoader(self, train_batchsize=None, test_batchsize=None):
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=train_batchsize, num_workers=self.numberwork, pin_memory=True, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=test_batchsize, num_workers=self.numberwork, pin_memory=True, shuffle=False)
        return self.trainloader, self.testloader


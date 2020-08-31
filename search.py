import sys
import random
import numpy as np
import json
import logging
import utils
import hashlib
import datetime
import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from models.cifardataset import CifarDataset
from models.genenetwork import GeneNetwork
from models.population import Population


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--gpu', type=int, default=-1, help='gpu device id, -1 denote use all gpus')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--numberworks', type=int, default=2, help='numberworks')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--datadir', type=str, default='dataset/cifar', help='location of the data corpus')
parser.add_argument('--init_learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--auxiliary', type=bool, default=True, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', type=bool, default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--warmup_epoch_first', type=int, default=50, help='train epochs of the first decision')
parser.add_argument('--warmup_epoch_others', type=int, default=3, help='train epochs of the left decisions')
parser.add_argument('--population_size', type=int, default=7, help='population size')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--reload_model', type=bool, default=False, help='reload models')
parser.add_argument('--dataset_name', type=str, default='cifar10', help='cifar10 or cifar100')
parser.add_argument('--cell_size', type=int, default=4, help='the number of the intermediate nodes of a cell')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
args = parser.parse_args()

#
if __name__ == '__main__':
    # logging
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    currenttime = datetime.datetime.now()
    logDir = 'recorddir/log/search_%s%s%s%s%s' % (
    currenttime.year, currenttime.month, currenttime.day, currenttime.hour, currenttime.minute)
    if not os.path.isdir(logDir):
        os.makedirs(logDir)
    fh = logging.FileHandler(os.path.join(logDir, 'search.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    # set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    cudnn.benchmark = True
    cudnn.enabled = True

    # best test accuracy
    best_acc = 0

    # decision number
    decision_number = (sum([(2 + i) for i in range(args.cell_size)]) + (args.cell_size-1))*2

    # Device
    GPUSTR = '' if args.gpu == -1 else ':%d' % (args.gpu)
    device = 'cuda' + GPUSTR if torch.cuda.is_available() else 'cpu'

    # dataloader
    cifarDataset = CifarDataset(args.datadir, bCutOut=args.cutout, dataset_name=args.dataset_name)
    search_train_dataLoader, search_valid_dataLoader, search_test_dataLoader = cifarDataset.getDataLoader(args.batch_size, args.batch_size, args.batch_size)

    # criterion optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    # checkpoint dir
    modelCheckPointDir = 'checkpoint/'
    if not os.path.isdir(modelCheckPointDir):
        os.makedirs(modelCheckPointDir)

    # checkpoint file name
    modelCheckPointName = 'firstwarmup_%d.pth'%(args.warmup_epoch_first)
    pretrainModelCheckPointPath = os.path.join(modelCheckPointDir, modelCheckPointName)

    # build models
    logging.info('==> Building models..')
    supernet = GeneNetwork(device, criterion, C=args.init_channels, stemC=args.init_channels * 3, num_classes=cifarDataset.datasetNumberClass, layerNum=args.layers, cellSize=args.cell_size, auxiliary=True)
    supernet = supernet.to(device)
    logging.info("param size = %fMB", utils.count_net_parameters(supernet))

    # if use multi gpus
    if args.gpu == -1:
        supernet = torch.nn.DataParallel(supernet)

    # optimizer
    supernetOptimizer = optim.SGD(supernet.parameters(), lr=args.init_learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    # scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(supernetOptimizer, args.warmup_epoch_first + decision_number*args.warmup_epoch_others)

    # define the population for search
    population = Population()
    logging.info('Population 0: %d', len(population.archList))


    # begin search
    for geneOpIndex in range(decision_number):
        # The number of train epochs of the first decision is greator than that of the others.
        if geneOpIndex == 0:
            tmp_warmup_epoch = args.warmup_epoch_first
        else:
            tmp_warmup_epoch = args.warmup_epoch_others

        # If you have already trained the models after first warming up, you can reload it to avoid too much time consuming.
        if args.reload_model and os.path.exists(pretrainModelCheckPointPath) and geneOpIndex == 0:
            logging.info('==> Resuming from checkpoint: %s', os.path.abspath(pretrainModelCheckPointPath))
            checkpoint = torch.load(pretrainModelCheckPointPath)
            supernet.load_state_dict(checkpoint['supernet'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            supernetOptimizer.load_state_dict(checkpoint['optimizer'])

            logging.info('reload models warmupepoch: %d', checkpoint['epoch'], checkpoint['acc'])
        else:
            # supernet training stage
            trainloss, traintop1 = population.trainsharespace(tmp_warmup_epoch, supernet, search_train_dataLoader, criterion, device,
                                                              supernetOptimizer, args.auxiliary, args.auxiliary_weight)
            logging.info('trainloss, traintop1: %f %f', trainloss, traintop1)

            # save the checkpoint after first warming up
            if geneOpIndex == 0:
                state = {
                    'supernet': supernet.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'optimizer': supernetOptimizer.state_dict(),
                    'epoch': tmp_warmup_epoch,
                    'acc': traintop1
                }
                torch.save(state, pretrainModelCheckPointPath)

            # adjust learning rate
            scheduler.step()

        # mutation
        newPopulationLen = population.mutation()
        logging.info('geneOpIndex: %d newPopulationLen: %d', geneOpIndex, newPopulationLen)

        if newPopulationLen == 0:
            # search over, output the rest population
            logging.info('geneOpIndex %d  gene over！', geneOpIndex)
            population.showAllArch()
            break
        elif newPopulationLen < args.population_size:
            # the search is still going on
            continue
        else:
            # nsga2 algorithm: population non-dominate sort, evaluate each individual's fitness and rand them
            logging.info('geneOpIndex %d  evaluate...', geneOpIndex)
            population.evaluate(supernet, search_valid_dataLoader, criterion, device)

            #nondomisort
            logging.info('geneOpIndex %d  nondomisort...', geneOpIndex)
            population.nondomisort()

            #crowding distance
            logging.info('geneOpIndex %d  crowding...', geneOpIndex)
            population.crowddis()

            #reject
            logging.info('geneOpIndex %d  reject...', geneOpIndex)
            population.reject(args.population_size)

            #show population info
            #populationJsoninfo, jsonStr = population.tostring()
            bestJsoninfo, bestJsonStr = population.selectBestArch()
            print('bestJsoninfo:', bestJsoninfo)
            logging.info('best acc arch: %s', bestJsonStr[0])
            logging.info('best latency arch: %s', bestJsonStr[1])


    #output the best arch info
    best_acc_arch = bestJsoninfo[0]
    print('best_acc_arch:', best_acc_arch)


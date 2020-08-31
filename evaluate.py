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
from models.fixedgenenetwork import FixedGeneNetwork

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--gpu', type=int, default=-1, help='gpu device id, -1 denote use all gpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--numberworks', type=int, default=2, help='numberworks')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--datadir', type=str, default='dataset/cifar', help='location of the data corpus')
parser.add_argument('--init_learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--auxiliary', type=bool, default=True, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', type=bool, default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--evaluate_epochs', type=int, default=600, help='train epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--reload_model', type=bool, default=False, help='reload models')
parser.add_argument('--dataset_name', type=str, default='cifar10', help='cifar10 or cifar100')
parser.add_argument('--cell_size', type=int, default=4, help='the number of the intermediate nodes of a cell')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--arch', type=str, default='HOENAS_A', help='which architecture to use')
args = parser.parse_args()


# warmup 50+5, 16, 8
HOENAS_A = {"normalOpArch": {"randomNum": 0, "op": [6, 4, 3, 4, 4, 3, 4, 4, 3, 4, 6, 0, 0, 0]},
        "reduceOpArch": {"randomNum": 0, "op": [5, 2, 4, 5, 4, 5, 1, 5, 5, 3, 1, 5, 4, 1]},
        "normalEdgeArch": {"randomNum": 1,
                           "edge": [{"randomType": False, "edge": [1, 1]}, {"randomType": False, "edge": [1, 0, 1]},
                                    {"randomType": False, "edge": [1, 1, 0, 0]},
                                    {"randomType": False, "edge": [0, 0, 0, 0, 1]}]},
        "reduceEdgeArch": {"randomNum": 1,
                           "edge": [{"randomType": False, "edge": [1, 1]}, {"randomType": False, "edge": [1, 0, 1]},
                                    {"randomType": False, "edge": [1, 1, 0, 0]},
                                    {"randomType": False, "edge": [0, 0, 0, 0, 1]}]}}

# # same model search
HOENAS_B = {"normalOpArch": {"randomNum": 0, "op": [5, 0, 3, 0, 6, 4, 0, 1, 4, 0, 0, 2, 3, 2]},
        "reduceOpArch": {"randomNum": 0, "op": [0, 4, 6, 6, 6, 4, 6, 3, 5, 0, 1, 3, 4, 5]},
        "normalEdgeArch": {"randomNum": 1,
                           "edge": [{"randomType": False, "edge": [1, 1]}, {"randomType": False, "edge": [1, 1, 0]},
                                    {"randomType": False, "edge": [0, 1, 0, 1]},
                                    {"randomType": False, "edge": [1, 1, 0, 0, 0]}]},
        "reduceEdgeArch": {"randomNum": 1,
                           "edge": [{"randomType": False, "edge": [1, 1]}, {"randomType": False, "edge": [1, 1, 0]},
                                    {"randomType": False, "edge": [0, 1, 0, 1]},
                                    {"randomType": False, "edge": [1, 1, 0, 0, 0]}]}
        }


# 推断过程, 测试accuracy
def evaluate_test_epoch(fixednet, testDataLoader, criterion, device):
    # 设置supernet训练模式
    fixednet.eval()
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    #
    for batch_idx, (testinputs, testtargets) in enumerate(testDataLoader):
        testinputs, testtargets = testinputs.to(device), testtargets.to(device)
        logits = fixednet(testinputs)
        loss = criterion(logits, testtargets)
        prec1, prec5 = utils.calAccuracy(logits, testtargets, topk=(1, 5))
        n = testinputs.size(0)
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)

        #break
    return objs.avg, top1.avg, top5.avg

# 训练过程, 优化网络架构
def evaluate_train_epoch(fixednet, trainDataLoader, criterion, device, optimizer, bAuxiliary, auxiliary_weight):
    # 设置supernet训练模式
    fixednet.train()

    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    # 获取训练数据
    for batch_idx, (traininputs, traintargets) in enumerate(trainDataLoader):
        traininputs, traintargets = traininputs.to(device), traintargets.to(device)

        # 训练supernet
        optimizer.zero_grad()
        logits, logits_aux = fixednet(traininputs)
        loss = criterion(logits, traintargets)
        if bAuxiliary:
            loss_aux = criterion(logits_aux, traintargets)
            loss += auxiliary_weight * loss_aux

        loss.backward()
        nn.utils.clip_grad_norm_(fixednet.parameters(), 5)
        optimizer.step()
        #
        prec1, prec5 = utils.calAccuracy(logits, traintargets, topk=(1, 5))
        tmpBatchSize = traininputs.size(0)
        objs.update(loss.data, tmpBatchSize)
        top1.update(prec1.data, tmpBatchSize)
        top5.update(prec5.data, tmpBatchSize)

    return objs.avg, top1.avg, top5.avg


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

    # 如果网络的输入数据维度或类型上变化不大，设置  torch.backends.cudnn.benchmark = true  可以增加运行效率
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
    evaluate_train_dataLoader, evaluate_test_dataLoader = cifarDataset.getFixDataLoader(args.batch_size, args.batch_size)

    # criterion optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    # checkpoint dir
    modelCheckPointDir = 'checkpoint/'
    if not os.path.isdir(modelCheckPointDir):
        os.makedirs(modelCheckPointDir)


    archInfo = eval(args.arch)
    jsonStr = json.dumps(archInfo)
    hashArchStr = hashlib.md5(jsonStr.encode('utf-8')).hexdigest()
    print('best_acc_arch hashArchStr:', hashArchStr)

    # checkpoint file name
    modelCheckPointName = 'evaluate_%s.pth' % (hashArchStr)
    evaluateModelCheckPointPath = os.path.join(modelCheckPointDir, modelCheckPointName)

    # writer, record models performance curve
    evaluate_writer = SummaryWriter(log_dir='recorddir/runs/evaluate_runs_%s%s%s%s%s' % (
        currenttime.year, currenttime.month, currenttime.day, currenttime.hour, currenttime.minute))

    # build models
    logging.info('==> Building models..')
    fixednet = FixedGeneNetwork(device, criterion, archInfo, C=args.init_channels, stemC=args.init_channels * 3,
                                layerNum=args.layers, cellSize=args.cell_size, auxiliary=True, num_classes=cifarDataset.datasetNumberClass)
    fixednet = fixednet.to(device)
    logging.info("param size = %fMB", utils.count_net_parameters(fixednet))
    # if use multi gpus
    if args.gpu == -1:
        fixednet = torch.nn.DataParallel(fixednet)

    # optimizer
    fixednetOptimizer = optim.SGD(fixednet.parameters(), lr=args.init_learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    # scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(fixednetOptimizer, args.evaluate_epochs)

    start_epoch = 0

    # reload model
    if args.reload_model:
        # 模型文件路径
        if os.path.exists(evaluateModelCheckPointPath):
            print('==> Resuming from checkpoint:', os.path.abspath(modelCheckPointDir))
            checkpoint = torch.load(evaluateModelCheckPointPath)
            fixednet.load_state_dict(checkpoint['fixednet'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            fixednetOptimizer.load_state_dict(checkpoint['optimizer'])

            start_epoch = checkpoint['epoch']
            acc = checkpoint['acc']
            print('reload model best_acc, startEpoch :', acc, start_epoch)

    # 开始darts训练
    for epoch in range(start_epoch, args.evaluate_epochs):
        # 时间调度员 记录 cosline lr
        lr = scheduler.get_lr()[0]  # get_last_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        fixednet.drop_path_prob = args.drop_path_prob * (epoch - start_epoch) / (args.evaluate_epochs - start_epoch)

        # 训练一个epoch
        with torch.autograd.set_detect_anomaly(True):
            trainLoss, trainTop1, trainTop5 = evaluate_train_epoch(fixednet, evaluate_train_dataLoader, criterion, device, fixednetOptimizer, args.auxiliary, args.auxiliary_weight)
        with torch.no_grad():
            testLoss, testTop1, testTop5 = evaluate_test_epoch(fixednet, evaluate_test_dataLoader, criterion, device)

        scheduler.step()
        evaluate_writer.add_scalars('scalar', {
            'trainLoss': trainLoss,
            'trainTop1': trainTop1,
            'trainTop5': trainTop5,
            'testLoss': testLoss,
            'testTop1': testTop1,
            'testTop5': testTop5
        }, epoch)

        #
        logging.info('HOENAS epoch:%03d', epoch)
        logging.info('trainloss:%e top1:%f top5:%f', trainLoss, trainTop1, trainTop5)
        logging.info('testloss:%e top1:%f top5:%f', testLoss, testTop1, testTop5)
        #
        print('Saving model...')
        state = {
            'fixednet': fixednet.state_dict(),
            'scheduler': scheduler.state_dict(),
            'optimizer': fixednetOptimizer.state_dict(),
            'epoch': epoch,
            'acc': testTop1
        }
        torch.save(state, evaluateModelCheckPointPath)

    evaluate_writer.close()
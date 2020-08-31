import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
from models.operations import Identity, drop_path, candidateNameList, candidateOpDict, ReLUConvBN, FactorizedReduce

from collections import namedtuple
import utils

Genotype = namedtuple('Genotype', 'normal reduce')


#
class MixedOp(nn.Module):
    def __init__(self, curChannel, stride, opActived, affine=True):
        super(MixedOp, self).__init__()

        candidateName = candidateNameList[opActived]
        #print(candidateNameList, candidateName)
        self.op = candidateOpDict[candidateName](curChannel, stride, affine)
        # pool 之后必接 BN
        # if 'pool' in candidateName:
        #     self.op = nn.Sequential(self.op, nn.BatchNorm2d(curChannel, affine=False))


    def forward(self, x):
        return self.op(x)



#
class Cell(nn.Module):
    def __init__(self, device, arch, prev_prev_C, prev_C, cur_C, bCurReduction, bPrevReduction, cellSize=4, newStateLen=4):
        super(Cell, self).__init__()
        #
        self.device = device
        self.cellSize = cellSize
        self.newStateLen = newStateLen

        self.bCurReduction = bCurReduction

        # 如果上一个Cell对图像尺寸进行了减半操作, 那么就将上上个Cell进行图像尺寸减半操作。 此时 preprocess0, 1 分别都是图像尺寸减半的操作
        #search 阶段 affine 都是False； valid阶段都是True，希望拟合得更好
        if bPrevReduction:
            self.preprocess0 = FactorizedReduce(prev_prev_C, cur_C, affine=True)
        else:
            self.preprocess0 = ReLUConvBN(prev_prev_C, cur_C, kernel_size=1, stride=1, padding=0, affine=True)

        self.preprocess1 = ReLUConvBN(prev_C, cur_C, kernel_size=1, stride=1, padding=0, affine=True)

        self.opArchInfo, self.edgeArchInfo = arch

        # 对于一个4size的Cell， 其中有4个节点。加上2个precess节点共6个。 对于每个节点正序连接的边都有一个MixedOp
        self._ops = nn.ModuleList()
        nodeIndex = 0
        for i in range(self.cellSize):
            for j in range(2 + i):
                #只选出edge标致为1的操作
                if self.edgeArchInfo['edge'][i]['edge'][j] == 1:
                    stride = 2 if bCurReduction and j < 2 else 1
                    # 将opactived的情况告诉
                    op = MixedOp(cur_C, stride, self.opArchInfo['op'][nodeIndex], affine=True)
    #                 print(bCurReduction, self.opArchInfo['op'][nodeIndex], sum(
    # p.numel() for p in op.parameters() if p.requires_grad))
    #                 print(op)
    #                 input()
                    self._ops.append(op)
                    #print(self.edgeArchInfo['edge'][i]['edge'][j])
                nodeIndex += 1

    # Darts中每个单元Cell有两个输入，以及一个权重。
    # weights 是一个2n + n(n-1)/2 长度的一个权重，代表了所有可能边的权重。 每一个都是一个relaxWeight。
    def forward(self, s0, s1, drop_path_prob=0):
        # 这两个s0 和 s1 的feature map 尺寸都是相同的
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        # 初始状态就是2个节点
        states = [s0, s1]

        offset = 0

        allOpIndex = 0
        for cellIndex in range(self.cellSize):
            # 将之前的状态，经过op, 融合为新的状态节点
            tmpStateList = []

            # 根据 self.edgeArchInfo 来决定每个cell的forward
            # pretrain阶段，从各个边中随机挑选2个训练
            nodeEdgeInfo = self.edgeArchInfo['edge'][cellIndex]
            edgeActivatedOfCurNode = np.array(nodeEdgeInfo['edge'])

            # 激活的edge索引
            activedEdgeIndexOfCurCell = np.argwhere(edgeActivatedOfCurNode == 1).reshape(-1)

            # 只计算激活的edge, 有能是1到2个 featuremap
            for activedEdgeIndex in activedEdgeIndexOfCurCell:
                #allOpIndex = offset + activedEdgeIndex
                tmpOp = self._ops[allOpIndex]
                allOpIndex += 1

                #print('cellIndex, opIndex:', cellIndex, activedEdgeIndex, tmpOp)
                #print(states[activedEdgeIndex].shape)
                tmpState = tmpOp(states[activedEdgeIndex])

                if self.training and drop_path_prob > 0.:
                    if not isinstance(tmpOp, Identity):
                        tmpState = drop_path(tmpState, drop_path_prob, self.device)

                tmpStateList.append(tmpState)

            # offset 迭代计算
            offset += len(states)

            # 只计算激活的edge
            newState = torch.stack(tmpStateList, dim=0).sum(dim=0)
            #print(tmpStateList[0].shape, tmpStateList[1].shape, newState.shape)

            states.append(newState)

        # 将states后面newStateLen个状态，在channel维度进行整合。 相当于就是原来的newStateLen倍数
        return torch.cat(states[-self.newStateLen:], dim=1)


# xception 2017 论文中, 如果batchsize=1，这里面的batchnorm会出错
class AuxiliaryHeadCIFAR(nn.Module):
    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),  # image size = 2 x 2
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        #print('AuxiliaryHeadCIFAR before:', x.shape)
        x = self.features(x)
        #print('AuxiliaryHeadCIFAR after:', x.shape)
        x = self.classifier(x.view(x.size(0), -1))
        return x


# GeneNetwork
class FixedGeneNetwork(nn.Module):
    def __init__(self, device, criterion, arch, C=36, stemC=36 * 3, channelFactor=4, num_classes=10, layerNum=8, cellSize=4,
                 auxiliary=True):
        super(FixedGeneNetwork, self).__init__()
        # 每个单元有多少个节点
        self.cellSize = cellSize
        #
        self.channelFactor = channelFactor

        self._criterion = criterion

        self.layerNum = layerNum
        self.drop_path_prob = 0

        #
        self._auxiliary = auxiliary

        # 初始主干网络
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=stemC, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stemC)
        )

        cur_C = stemC

        #
        self.cells = nn.ModuleList()
        #
        prev_prev_C, prev_C, cur_C = cur_C, cur_C, C

        bPrevReduction = False
        for layerIndex in range(self.layerNum):
            # 在 layerNum//3, 2*layerNum//3 这两个节点，将图像尺寸减半
            if layerIndex in [self.layerNum // 3, 2 * self.layerNum // 3]:
                cur_C *= 2
                bCurReduction = True
                cell = Cell(device, [arch['reduceOpArch'], arch['reduceEdgeArch']], prev_prev_C, prev_C, cur_C, bCurReduction, bPrevReduction, cellSize=4)
            else:
                bCurReduction = False
                cell = Cell(device, [arch['normalOpArch'], arch['normalEdgeArch']], prev_prev_C, prev_C, cur_C, bCurReduction, bPrevReduction, cellSize=4)


            print('layerIndex prev_prev_C, prev_C, cur_C:', layerIndex, prev_prev_C, prev_C, cur_C)
            self.cells.append(cell)

            bPrevReduction = bCurReduction
            prev_prev_C, prev_C = prev_C, channelFactor * cur_C

            if layerIndex == 2 * layerNum // 3:
                to_auxiliary_C = prev_C

        # last
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(prev_C, num_classes)

        #
        if auxiliary:
            #print('auxiliary:', to_auxiliary_C, num_classes)
            self.auxiliary_head = AuxiliaryHeadCIFAR(to_auxiliary_C, num_classes)
    #
    def forward(self, input):
        # 初始的两个节点 就是 主干输出， 3 -> 48
        s0 = s1 = self.stem(input)
        for cellIndex, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)

            #
            if cellIndex == 2 * self.layerNum // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)

        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))  # batchSize, 10

        if self._auxiliary and self.training:
            return logits, logits_aux
        else:
            return logits

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

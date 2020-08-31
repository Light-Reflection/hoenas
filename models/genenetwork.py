import torch
import torch.nn as nn
import numpy as np
from models.operations import Identity, drop_path, candidateNameList, candidateOpDict, ReLUConvBN, FactorizedReduce

import utils

#MixedOp
class MixedOp(nn.Module):
    def __init__(self, curChannel, stride):
        super(MixedOp, self).__init__()

        self._ops = nn.ModuleList()
        for candidateName in candidateNameList:
            op = candidateOpDict[candidateName](curChannel, stride, False)
            # BN is added after pool
            if 'pool' in candidateName:
                op = nn.Sequential(op, nn.BatchNorm2d(curChannel, affine=False))
            self._ops.append(op)

        self.pretrainStage = True

    def setOpArch(self, opActived):
        self.activedOpIndex = opActived

    def forward(self, x):
        # -1 is random code
        if self.activedOpIndex == -1:
            activedOpIndexOfCurNode = utils.combineSample(len(self._ops), 1)
            a = [self._ops[activedOpIndex](x) for activedOpIndex in activedOpIndexOfCurNode]
            return sum(a)
        else:
            a = self._ops[self.activedOpIndex](x)
            return a


# Cell
class Cell(nn.Module):
    def __init__(self, device, prev_prev_C, prev_C, cur_C, bCurReduction, bPrevReduction, cellSize=4, newStateLen=4):
        super(Cell, self).__init__()
        self.device = device

        #
        self.cellSize = cellSize
        self.newStateLen = newStateLen

        self.bCurReduction = bCurReduction

        #
        if bPrevReduction:
            self.preprocess0 = FactorizedReduce(prev_prev_C, cur_C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(prev_prev_C, cur_C, kernel_size=1, stride=1, padding=0, affine=False)

        self.preprocess1 = ReLUConvBN(prev_C, cur_C, kernel_size=1, stride=1, padding=0, affine=False)

        #
        self._ops = nn.ModuleList()
        nodeIndex = 0
        for i in range(self.cellSize):
            for j in range(2 + i):
                stride = 2 if bCurReduction and j < 2 else 1
                #
                op = MixedOp(cur_C, stride)
                self._ops.append(op)
                nodeIndex += 1

    #
    def setCellArch(self, arch):
        self.opArchInfo, self.edgeArchInfo = arch

        nodeIndex = 0
        for op in self._ops:
            op.setOpArch(self.opArchInfo['op'][nodeIndex])
            nodeIndex += 1

    # relaxWeight : 2n + n(n-1)/2
    def forward(self, s0, s1, drop_path_prob=0):
        # s0 and s1's feature map are same
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        # all nodes: feature maps
        states = [s0, s1]

        offset = 0
        for cellIndex in range(self.cellSize):
            # the feature map after operation computation
            tmpStateList = []

            for opIndex, state in enumerate(states):
                # every op here is a mixedop
                allOpIndex = offset + opIndex
                tmpOp = self._ops[allOpIndex]
                tmpState = tmpOp(state)

                if self.training and drop_path_prob > 0.:
                    if not isinstance(tmpOp, Identity):
                        tmpState = drop_path(tmpState, drop_path_prob, self.device)

                tmpStateList.append(tmpState)

            offset += len(states)

            # the depthwise concatenation is ruled by edge activation
            nodeEdgeInfo = self.edgeArchInfo['edge'][cellIndex]
            edgeOfCellRandomType = nodeEdgeInfo['randomType']
            if edgeOfCellRandomType:
                activedEdgeIndexOfCurCell = utils.combineSample(len(tmpStateList), 2)

                # index by edge activation
                newState = sum([tmpStateList[activedEdgeIndex] for activedEdgeIndex in activedEdgeIndexOfCurCell])
                states.append(newState)
            else:
                edgeActivatedOfCurNode = np.array(nodeEdgeInfo['edge'])

                # index by edge activation
                activedEdgeIndexOfCurCell = np.argwhere(edgeActivatedOfCurNode == 1).reshape(-1)

                newState = sum([tmpStateList[activedEdgeIndex] for activedEdgeIndex in activedEdgeIndexOfCurCell])
                states.append(newState)

        # depthwise concatenation of the last 4 states
        return torch.cat(states[-self.newStateLen:], dim=1)


#
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
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


# supernet
class GeneNetwork(nn.Module):
    def __init__(self, device, criterion, C=36, stemC=36 * 3, channelFactor=4, num_classes=10, layerNum=8, cellSize=4,
                 auxiliary=True):
        super(GeneNetwork, self).__init__()
        self.cellSize = cellSize
        self.channelFactor = channelFactor
        self._criterion = criterion
        self.layerNum = layerNum
        self.drop_path_prob = 0
        self._auxiliary = auxiliary

        # 初始主干网络
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=stemC, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stemC)
        )

        cur_C = stemC

        self.cells = nn.ModuleList()
        prev_prev_C, prev_C, cur_C = cur_C, cur_C, C

        bPrevReduction = False
        for layerIndex in range(self.layerNum):
            # the reduction cell is located at layerNum//3, 2*layerNum//3
            if layerIndex in [self.layerNum // 3, 2 * self.layerNum // 3]:
                cur_C *= 2
                bCurReduction = True
            else:
                bCurReduction = False

            cell = Cell(device, prev_prev_C, prev_C, cur_C, bCurReduction, bPrevReduction, cellSize=4)

            print('layerIndex prev_prev_C, prev_C, cur_C:', layerIndex, prev_prev_C, prev_C, cur_C)
            self.cells.append(cell)

            bPrevReduction = bCurReduction
            prev_prev_C, prev_C = prev_C, channelFactor * cur_C

            if layerIndex == 2 * layerNum // 3:
                to_auxiliary_C = prev_C

        # last
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(prev_C, num_classes)

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(to_auxiliary_C, num_classes)

    #
    def setarch(self, arch):
        layerIndex = 0
        for cell in self.cells:
            if layerIndex in [self.layerNum // 3, 2 * self.layerNum // 3]:
                cell.setCellArch([arch.reduceOpArch, arch.reduceEdgeArch])
            else:
                cell.setCellArch([arch.normalOpArch, arch.normalEdgeArch])

            layerIndex += 1

    #
    def forward(self, input):
        # stem： channel 3 -> 48
        s0 = s1 = self.stem(input)
        for cellIndex, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)

            #
            if cellIndex == 2 * self.layerNum // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)

        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))

        if self._auxiliary and self.training:
            return logits, logits_aux
        else:
            return logits

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

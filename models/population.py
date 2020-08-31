import numpy as np
from utils import AvgrageMeter, calAccuracy, combineSample
from models.archinfo import ArchInfo
from models.operations import candidateNameList
import time
import torch
import json
import torch.nn as nn

# population of architectures
class Population(object):
    #
    def __init__(self):
        self.archList = []

        #the first population generation is a desicion of the last operation computaion
        for opindex in range(len(candidateNameList)):
            arch = ArchInfo()
            arch.normalOpArch['randomNum'] = arch.normalOpArch['randomNum'] - 1
            changeLastNodeIndex = arch.normalOpArch['randomNum']
            arch.normalOpArch['op'][changeLastNodeIndex] = opindex
            self.archList.append(arch)


    # evaluate each individual
    def evaluate(self, network, dataloader, criterion, device, test_flag=False):
        #
        archIndex = 0
        for arch in self.archList:
            if test_flag:  #fast
                arch.performance['testacc'] = np.random.rand()
                arch.performance['latency'] = -np.random.rand()
                continue

            #print('population evaluate archIndex:', archIndex)
            archIndex+=1
            # setarch
            if hasattr(network, 'module'):
                network.module.setarch(arch)
            else:
                network.setarch(arch)

            with torch.no_grad():
                loss, top1acc, top5acc, latency = self.testepoch(network, dataloader, criterion, device)

            arch.performance['testacc'] = top1acc.data.cpu().item()
            arch.performance['latency'] = -latency

    # dominate compare
    def dominate(self, archID1, archID2):
        if self.archList[archID1].performance['testacc'] > self.archList[archID2].performance['testacc'] \
                and self.archList[archID1].performance['latency'] < self.archList[archID2].performance['latency']:
            return True
        else:
            return False

    # nondomisort
    def nondomisort(self):
        # population size
        populationLen = len(self.archList)

        # record the individuals dominated by every individual
        dominateListOfEachArch = [[] for archIndex in range(populationLen)]
        # record the number of the individuals who dominate it
        dominatedNumOfEachArch = np.zeros(populationLen)
        # record the rank of every individual
        RANK = np.zeros(populationLen)

        # frontList
        frontList = [[]]

        # nondomisort
        for archIndex in range(populationLen):
            for compareArchIndex in range(populationLen):
                if archIndex != compareArchIndex:
                    if self.dominate(archIndex, compareArchIndex):
                        dominateListOfEachArch[archIndex].append(compareArchIndex)
                    elif self.dominate(compareArchIndex, archIndex):
                        dominatedNumOfEachArch[archIndex] += 1

            # If an individual is not dominated by everyone, it's rank is 1.
            if dominatedNumOfEachArch[archIndex] == 0:
                RANK[archIndex] = 1
                frontList[0].append(archIndex)

        Q = [1]
        FIndex = 0
        while len(Q) != 0:
            Q = []
            # for the individuals of the upper front
            for archIndex in frontList[FIndex]:
                # weakArchIndex domiated by architecture 'archIndex'
                for weakArchIndex in dominateListOfEachArch[archIndex]:
                    # remove the upper influence
                    dominatedNumOfEachArch[weakArchIndex] = dominatedNumOfEachArch[weakArchIndex] - 1
                    # rank every individual except the upper front
                    if dominatedNumOfEachArch[weakArchIndex] == 0:
                        RANK[weakArchIndex] = FIndex + 1
                        Q.append(weakArchIndex)

            frontList.append(Q)
            FIndex+=1

        self.frontList = frontList
        self.RANK = RANK
        print(self.frontList)

    # crowddis
    def crowddis(self):
        # 2 metrics, acc and lantency
        populationLen = len(self.archList)
        archAndObjectArray = np.array(
            [[archIndex, self.archList[archIndex].performance['testacc'], self.archList[archIndex].performance['latency']] for archIndex in range(populationLen)]
        )
        #
        archAndObjectArray = archAndObjectArray.transpose()
        #print('population crowddis archAndObjectArray:', archAndObjectArray)

        crowdDisList = [0 for i in range(populationLen)]
        for objectIndex in range(2):
            sortedArchIndexList = np.argsort(archAndObjectArray[objectIndex+1, :])
            # force save the best and the worst
            crowdDisList[sortedArchIndexList[0]] = 1.0
            crowdDisList[sortedArchIndexList[-1]] = 1.0

            objectMax = archAndObjectArray[objectIndex+1, sortedArchIndexList[-1]]
            objectMin = archAndObjectArray[objectIndex+1, sortedArchIndexList[0]]
            # print('sortedArchIndexList:', sortedArchIndexList)
            # print('archAndObjectArray:', archAndObjectArray)
            for sortedArchIndex in range(1, len(sortedArchIndexList)-1):
                realIndex = sortedArchIndexList[sortedArchIndex]
                smallIndex = sortedArchIndexList[sortedArchIndex-1]
                bigIndex = sortedArchIndexList[sortedArchIndex - 1]
                crowdDisList[realIndex] = crowdDisList[realIndex] + (archAndObjectArray[objectIndex+1, bigIndex] - archAndObjectArray[objectIndex+1, smallIndex])*1.0/(objectMax - objectMin + 1e-10)

        self.crowdDisList = crowdDisList

    #reject
    def reject(self, MAX_P):
        newPopulation = []
        for front in self.frontList:
            #print('reject:', newPopulation, front)
            if (len(newPopulation) + len(front)) > MAX_P:
                #sort the front
                archAndDisArray = np.array(
                    [[archIndex, self.crowdDisList[archIndex]] for archIndex in front]
                )
                #
                archAndDisArray = archAndDisArray.transpose()
                #print('archAndDisArray:', archAndDisArray)
                sortedArchIndexList = np.argsort(archAndDisArray[1, :])
                #print('sortedArchIndexList:', sortedArchIndexList, len(newPopulation)-MAX_P)
                outputArchIndexList = sortedArchIndexList[len(newPopulation)-MAX_P:].tolist()
                newPopulation += outputArchIndexList
                #print('outputArchIndexList:', outputArchIndexList)

                break
            else:
                newPopulation += front

        newArchList = []
        for archIndex in newPopulation:
            newArchList.append(self.archList[archIndex])

        self.archList = newArchList

    #
    def mutation(self):
        newArchList = []
        for arch in self.archList:
            tmpArchSons = arch.mutation()
            newArchList += tmpArchSons

        newPopulationLen = len(newArchList)
        if newPopulationLen > 0:
            self.archList = newArchList

        return newPopulationLen

    #
    def tostring(self):
        populationJsoninfo = []
        for arch in self.archList:
            archJsonInfo, jsonStr = arch.tostring()
            populationJsoninfo.append(archJsonInfo)

        jsonStr = json.dumps(populationJsoninfo)
        return populationJsoninfo, jsonStr

    #  selectBestArch
    def selectBestArch(self):
        populationLen = len(self.archList)
        archAndObjectArray = np.array(
            [[archIndex, self.archList[archIndex].performance['testacc'],
              self.archList[archIndex].performance['latency']] for archIndex in range(populationLen)]
        )

        archAndObjectArray = archAndObjectArray.transpose()

        bestArchIndexs = []
        for objectIndex in range(2):
            sortedArchIndexList = np.argsort(archAndObjectArray[objectIndex + 1, :])
            bestArchIndexs.append(sortedArchIndexList[-1])

        #
        bestJsoninfo = []
        bestJsonStr = []
        for archIndex in bestArchIndexs:
            arch = self.archList[archIndex]
            archJsonInfo, jsonStr = arch.tostring()
            bestJsoninfo.append(archJsonInfo)

            jsonStr = json.dumps(archJsonInfo)
            bestJsonStr.append(jsonStr)
        return bestJsoninfo, bestJsonStr

    #  showAllArch
    def showAllArch(self):
        AllArchInfo = []
        AllJsonStr =[]
        for arch in self.archList:
            archJsonInfo, jsonStr = arch.tostring()
            AllArchInfo.append(archJsonInfo)

            jsonStr = json.dumps(archJsonInfo)
            AllJsonStr.append(jsonStr)
        return AllArchInfo, AllJsonStr

    # getShareSpaceArch
    def getShareSpaceArch(self):
        # random select one from the archList
        rand_arch_index = combineSample(len(self.archList), 1)[0]
        trivalArch = self.archList[rand_arch_index]

        if trivalArch.normalOpArch['randomNum'] >= 0:
            newArchInfo = trivalArch.copy()
            newArchInfo.normalOpArch['op'][newArchInfo.normalOpArch['randomNum']] = -1
            newArchInfo.normalOpArch['randomNum'] = newArchInfo.normalOpArch['randomNum'] + 1
        elif trivalArch.reduceOpArch['randomNum'] >= 0:
            newArchInfo = trivalArch.copy()
            newArchInfo.reduceOpArch['op'][newArchInfo.reduceOpArch['randomNum']] = -1
            newArchInfo.reduceOpArch['randomNum'] = newArchInfo.reduceOpArch['randomNum'] + 1
        elif trivalArch.normalEdgeArch['randomNum'] >= 1:
            newArchInfo = trivalArch.copy()
            newArchInfo.normalEdgeArch['randomNum'] = newArchInfo.normalEdgeArch['randomNum'] + 1
            edgeLen = newArchInfo.normalEdgeArch['randomNum'] + 1
            newArchInfo.normalEdgeArch['edge'][newArchInfo.normalEdgeArch['randomNum']-1]['randomType'] = True
            newArchInfo.normalEdgeArch['edge'][newArchInfo.normalEdgeArch['randomNum']-1]['edge'] = -1*np.ones(edgeLen, dtype=np.int8)
        elif trivalArch.reduceEdgeArch['randomNum'] >= 1:
            newArchInfo = trivalArch.copy()
            newArchInfo.reduceEdgeArch['randomNum'] = newArchInfo.reduceEdgeArch['randomNum'] + 1
            edgeLen = newArchInfo.reduceEdgeArch['randomNum'] + 1
            newArchInfo.reduceEdgeArch['edge'][newArchInfo.reduceEdgeArch['randomNum'] - 1]['randomType'] = True
            newArchInfo.reduceEdgeArch['edge'][newArchInfo.reduceEdgeArch['randomNum'] - 1]['edge'] = -1 * np.ones(
                edgeLen, dtype=np.int8)
        else:
            print('出现意外中的情况')
            newArchInfo = trivalArch

        return newArchInfo


    # testepoch
    def testepoch(self, fixednet, testDataLoader, criterion, device):
        fixednet.eval()

        objs = AvgrageMeter()
        top1 = AvgrageMeter()
        top5 = AvgrageMeter()

        begin_time = time.time()
        for batch_idx, (testinputs, testtargets) in enumerate(testDataLoader):
            testinputs, testtargets = testinputs.to(device), testtargets.to(device)
            logits = fixednet(testinputs)
            loss = criterion(logits, testtargets)
            prec1, prec5 = calAccuracy(logits, testtargets, topk=(1, 5))
            n = testinputs.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)
            #break # fast

        latency = time.time() - begin_time
        return objs.avg, top1.avg, top5.avg, latency

    # search_train_epoch
    def search_train_epoch(self, fixednet, trainDataLoader, criterion, device, genenetOptimizer, bAuxiliary, auxiliary_weight):
        fixednet.train()

        objs = AvgrageMeter()
        top1 = AvgrageMeter()
        top5 = AvgrageMeter()

        #
        for batch_idx, (traininputs, traintargets) in enumerate(trainDataLoader):
            traininputs, traintargets = traininputs.to(device), traintargets.to(device)

            #
            shareArch = self.getShareSpaceArch()
            if hasattr(fixednet, 'module'):
                fixednet.module.setarch(shareArch)
            else:
                fixednet.setarch(shareArch)

            #
            genenetOptimizer.zero_grad()
            logits, logits_aux = fixednet(traininputs)
            loss = criterion(logits, traintargets)
            if bAuxiliary:
                loss_aux = criterion(logits_aux, traintargets)
                loss += auxiliary_weight * loss_aux

            loss.backward()
            nn.utils.clip_grad_norm_(fixednet.parameters(), 5)
            genenetOptimizer.step()
            #
            prec1, prec5 = calAccuracy(logits, traintargets, topk=(1, 5))
            tmpBatchSize = traininputs.size(0)
            objs.update(loss.data, tmpBatchSize)
            top1.update(prec1.data, tmpBatchSize)
            top5.update(prec5.data, tmpBatchSize)
            #break # fast

        return objs.avg, top1.avg, top5.avg

    # trainsharespace
    def trainsharespace(self, GENE_TRAIN_EPOCH, genenet, trainDataLoader, criterion, device, genenetOptimizer, bAuxiliary, auxiliary_weight, test_flag=False):
        if test_flag:
            return 0, 0  # fast

        for trainEpoch in range(GENE_TRAIN_EPOCH):
            trainloss, traintop1, traintop5 = self.search_train_epoch(genenet, trainDataLoader, criterion, device, genenetOptimizer, bAuxiliary, auxiliary_weight)
            print('trainEpoch population trainsharespace trainloss, traintop1, traintop5:', trainEpoch, trainloss.data.cpu().item(), traintop1.data.cpu().item(), traintop5.data.cpu().item())

        return trainloss.data.cpu().item(), traintop1.data.cpu().item()

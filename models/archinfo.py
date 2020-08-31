import numpy as np
import utils
import json
from models.operations import candidateNameList


# arch is represented by 2 parts: op and edge.
class ArchInfo(object):
    #
    def __init__(self):
        self.NODE_SIZE = 14
        self.OP_SIZE = len(candidateNameList)
        self.CELL_SIZE = 4
        self.EDGE_SIZE = 14

        #op info, -1 is the random code.
        self.normalOpArch = {
            'randomNum': self.NODE_SIZE,
            'op': -1*np.ones(self.NODE_SIZE, dtype=np.int8)
        }
        self.reduceOpArch = {
            'randomNum': self.NODE_SIZE,
            'op': -1*np.ones(self.NODE_SIZE, dtype=np.int8)
        }

        # edge info
        self.normalEdgeArch = {
            'randomNum': self.CELL_SIZE,
            'edge': [],
        }
        self.reduceEdgeArch = {
            'randomNum': self.CELL_SIZE,
            'edge': [],
        }
        tmpEdgeSize = 2
        for cellIndex in range(self.CELL_SIZE):
            if cellIndex == 0:
                nodeEdgeInfo = {
                    'randomType': False,
                    'edge': np.ones(tmpEdgeSize, dtype=np.int8)
                }
            else:
                nodeEdgeInfo = {
                    'randomType': True,
                    'edge': -1*np.ones(tmpEdgeSize, dtype=np.int8)
                }
            self.normalEdgeArch['edge'].append(nodeEdgeInfo.copy())
            self.reduceEdgeArch['edge'].append(nodeEdgeInfo.copy())
            tmpEdgeSize += 1

        self.performance = {
            'testacc': 0,
            'latency': 0,
            'sparsity': 0,
            'accdrop': 0
        }

    # json to string
    def tostring(self):
        jsonInfo = {
            'normalOpArch': {
                'randomNum': self.normalOpArch['randomNum'],
                'op': self.normalOpArch['op'].tolist()
            },
            'reduceOpArch': {
                'randomNum': self.reduceOpArch['randomNum'],
                'op': self.reduceOpArch['op'].tolist()
            },
            'normalEdgeArch': {
                'randomNum': self.normalEdgeArch['randomNum'],
                'edge': [{
                    'randomType': edgeinfo['randomType'],
                    'edge': edgeinfo['edge'].tolist()
                } for edgeinfo in self.normalEdgeArch['edge']]
            },
            'reduceEdgeArch': {
                'randomNum': self.reduceEdgeArch['randomNum'],
                'edge': [{
                    'randomType': edgeinfo['randomType'],
                    'edge': edgeinfo['edge'].tolist()
                } for edgeinfo in self.reduceEdgeArch['edge']]
            },
            'performance': self.performance,
        }
        jsonStr = json.dumps(jsonInfo)
        return jsonInfo, jsonStr

    # copy
    def copy(self):
        newArchInfo = ArchInfo()
        newArchInfo.normalOpArch['randomNum'] = self.normalOpArch['randomNum']
        newArchInfo.normalOpArch['op'] = self.normalOpArch['op'].copy()

        newArchInfo.reduceOpArch['randomNum'] = self.reduceOpArch['randomNum']
        newArchInfo.reduceOpArch['op'] = self.reduceOpArch['op'].copy()

        newArchInfo.normalEdgeArch['randomNum'] = self.normalEdgeArch['randomNum']
        for edgeindex in range(len(newArchInfo.normalEdgeArch['edge'])):
            newArchInfo.normalEdgeArch['edge'][edgeindex]['randomType'] = self.normalEdgeArch['edge'][edgeindex]['randomType']
            newArchInfo.normalEdgeArch['edge'][edgeindex]['edge'] = self.normalEdgeArch['edge'][edgeindex]['edge'].copy()

        newArchInfo.reduceEdgeArch['randomNum'] = self.reduceEdgeArch['randomNum']
        for edgeindex in range(len(newArchInfo.reduceEdgeArch['edge'])):
            newArchInfo.reduceEdgeArch['edge'][edgeindex]['randomType'] = self.reduceEdgeArch['edge'][edgeindex]['randomType']
            newArchInfo.reduceEdgeArch['edge'][edgeindex]['edge'] = self.reduceEdgeArch['edge'][edgeindex]['edge'].copy()

        if self.reduceEdgeArch['edge'][1]['edge'][0] != -1 :
            pass

        return newArchInfo


    # archi mutation,  in order "normalOpArch, reduceOpArch, normalEdgeArch, reduceEdgeArch"
    def mutation(self):
        newArchList = []
        if self.normalOpArch['randomNum'] > 0:
            for opIndex in range(self.OP_SIZE):
                newArchInfo = self.copy()
                newArchInfo.normalOpArch['randomNum'] = newArchInfo.normalOpArch['randomNum'] - 1
                newArchInfo.normalOpArch['op'][newArchInfo.normalOpArch['randomNum']] = opIndex
                newArchList.append(newArchInfo)
        elif self.reduceOpArch['randomNum'] > 0:
            for opIndex in range(self.OP_SIZE):
                newArchInfo = self.copy()
                newArchInfo.reduceOpArch['randomNum'] = newArchInfo.reduceOpArch['randomNum'] - 1
                newArchInfo.reduceOpArch['op'][newArchInfo.reduceOpArch['randomNum']] = opIndex
                newArchList.append(newArchInfo)
        elif self.normalEdgeArch['randomNum'] > 1:
            edgeLen = self.normalEdgeArch['randomNum'] + 1
            combinList = utils.combination(range(edgeLen), 2)
            for combineEdge in combinList:
                newArchInfo = self.copy()
                newArchInfo.normalEdgeArch['randomNum'] = newArchInfo.normalEdgeArch['randomNum'] - 1
                newArchInfo.normalEdgeArch['edge'][newArchInfo.normalEdgeArch['randomNum']]['randomType'] = False
                newArchInfo.normalEdgeArch['edge'][newArchInfo.normalEdgeArch['randomNum']]['edge'] = np.array([1 if edgeIndex in combineEdge else 0 for edgeIndex in range(edgeLen)], dtype=np.int8)
                newArchList.append(newArchInfo)

            # for comparison to darts.  darts have 'none' operation
            combinList = utils.combination(range(edgeLen), 1)
            for combineEdge in combinList:
                newArchInfo = self.copy()
                newArchInfo.normalEdgeArch['randomNum'] = newArchInfo.normalEdgeArch['randomNum'] - 1
                newArchInfo.normalEdgeArch['edge'][newArchInfo.normalEdgeArch['randomNum']]['randomType'] = False
                newArchInfo.normalEdgeArch['edge'][newArchInfo.normalEdgeArch['randomNum']]['edge'] = np.array([1 if edgeIndex in combineEdge else 0 for edgeIndex in range(edgeLen)], dtype=np.int8)
                newArchList.append(newArchInfo)

        elif self.reduceEdgeArch['randomNum'] > 1:
            edgeLen = self.reduceEdgeArch['randomNum']+1

            combinList = utils.combination(range(edgeLen), 2)
            for combineEdge in combinList:
                newArchInfo = self.copy()
                newArchInfo.reduceEdgeArch['randomNum'] = newArchInfo.reduceEdgeArch['randomNum'] - 1
                newArchInfo.reduceEdgeArch['edge'][newArchInfo.reduceEdgeArch['randomNum']]['randomType'] = False
                newArchInfo.reduceEdgeArch['edge'][newArchInfo.reduceEdgeArch['randomNum']]['edge'] = np.array([1 if edgeIndex in combineEdge else 0 for edgeIndex in range(edgeLen)], dtype=np.int8)
                newArchList.append(newArchInfo)

            combinList = utils.combination(range(edgeLen), 1)
            for combineEdge in combinList:
                newArchInfo = self.copy()
                newArchInfo.reduceEdgeArch['randomNum'] = newArchInfo.reduceEdgeArch['randomNum'] - 1
                newArchInfo.reduceEdgeArch['edge'][newArchInfo.reduceEdgeArch['randomNum']]['randomType'] = False
                newArchInfo.reduceEdgeArch['edge'][newArchInfo.reduceEdgeArch['randomNum']]['edge'] = np.array([1 if edgeIndex in combineEdge else 0 for edgeIndex in range(edgeLen)], dtype=np.int8)
                newArchList.append(newArchInfo)
        else:
            return []

        return newArchList
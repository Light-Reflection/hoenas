import sys
import time
import numpy as np
import torch

#_, term_width = os.popen('stty size', 'r').read().split()
term_width = 97#  int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

# count_net_parameters
def count_net_parameters(model):
    # for name, v in models.named_parameters():
    #     print(name, np.prod(v.size()))
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


# AvgrageMeter
class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


# calAccuracy
def calAccuracy(output, target, topk=(1,)):
    # output, target: torch.Size([96, 10]) torch.Size([96])

    maxk = max(topk)
    #int

    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    # _, pred: torch.Size([96, 5]) torch.Size([96, 5])

    #print('output, target, pred:', output.shape, target.shape, pred.shape)
    pred = pred.t()
    # pred: torch.Size([5, 96])

    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # correct: torch.Size([5, 96])

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

# combineSample, input: 3, 2  output:[0, 2]
def combineSample(maxNum, sampleNum):
    pool = np.r_[0:maxNum]

    ret = []
    for i in range(sampleNum):
        z = np.random.choice(pool, 1)
        index = np.where(pool == z)
        pool = np.delete(pool, index)
        ret.append(z[0])
    return ret

# combineSampleRetList, input: 3, 2  output:[0, 2]
def combineSampleRetList(maxNum, sampleNum):
    retFlagList = np.zeros(maxNum)

    pool = np.r_[0:maxNum]
    for i in range(sampleNum):
        z = np.random.choice(pool, 1)
        index = np.where(pool == z)
        pool = np.delete(pool, index)
        retFlagList[z[0]] = 1

    return retFlagList

#GenerateArch
def GenerateArch(testArchNum = 10):
    def SampleArch(sampleNum):
        nodeNum = 14
        opNum = 6

        choice = np.random.choice(opNum, sampleNum * nodeNum)
        choice = np.eye(opNum)[choice]
        choice = np.array(choice, dtype=np.int8).reshape(sampleNum, nodeNum, opNum)
        return choice

    def SampleEdge(sampleNum):
        cellSize = 4
        nodeNum = 14

        ret = []
        for sampleIndex in range(sampleNum):
            start = 0
            sampleList = np.zeros(nodeNum)
            for cellIndex in range(cellSize):
                edgeNum = 2 + cellIndex

                # 从edgeNum中选出2条
                choice = combineSample(edgeNum, 2)
                choice = [i + start for i in choice]

                start = start + edgeNum
                sampleList[choice] = 1

            ret.append(sampleList)

        ret = np.array(ret, dtype=np.int8)
        return ret

    allNormalOp = SampleArch(testArchNum)
    allReduceOp = SampleArch(testArchNum)

    allNormalEdge = SampleEdge(testArchNum)
    allReduceEdge = SampleEdge(testArchNum)
    return allNormalOp, allReduceOp, allNormalEdge, allReduceEdge


#combination
def combination(nums, k):
    ans = [[]]
    for i in range(k):
        ans = [pre + [sub]
               for pre in ans for sub in nums if i == 0 or sub > pre[-1]]
    return ans



# mixup
def mixup_data(x, y, device, alpha=2.0):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)





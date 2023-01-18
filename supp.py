import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def findforclosesta(var, q):
    index = 0
    t = 0
    for i in q:
        if np.abs(i-var) <= np.abs(q[index] - var):
            index = t
        t += 1
    a = np.arange(1., 15., .10)
    return a[index]


def checkvariance(var, K, numdata):
    np.random.seed(1)
    N = 100
    x = np.arange(2, N + 2)
    q = []
    for a in np.arange(1., 15., .1):
        weights = x ** (-a)
        weights /= weights.sum()
        bounded_zipf = stats.rv_discrete(name='bounded_zipf', values=(x, weights))
        sample = bounded_zipf.rvs(size=K)
        sample = sample / sample.sum() * numdata
        sumupvalue = numdata
        num_items = sample
        num_items = np.around(num_items)
        sum = num_items.sum()
        rndValues = np.random.choice(K, 10 * K, replace=True)
        t = 0
        if sum - sumupvalue > 0:
            # deduct
            count = sum - sumupvalue
            while t <= count:
                if num_items[rndValues[t]] - 1 > 0:
                    num_items[rndValues[rndValues[t]]] -= 1
                    t += 1
                else:
                    count += 1
                    t += 1
        else:
            # addition
            count = sumupvalue - sum
            while t < count:
                num_items[rndValues[rndValues[t]]] += 1
                t += 1
        q.append(int(np.var(num_items, ddof=0)))
    a = findforclosesta(var, q)
    return a

def getnumites(a, K, numdata):
    np.random.seed(1)
    N = 100
    x = np.arange(2, N + 2)
    weights = x ** (-a)
    weights /= weights.sum()
    bounded_zipf = stats.rv_discrete(name='bounded_zipf', values=(x, weights))
    sample = bounded_zipf.rvs(size=K)
    sample = sample / sample.sum() * numdata
    sumupvalue = numdata
    num_items = sample
    num_items = np.around(num_items)
    sum = num_items.sum()
    rndValues = np.random.choice(K, 10*K, replace=True)
    t = 0
    if sum - sumupvalue > 0:
        # deduct
        count = sum - sumupvalue
        while t <= count:
            if num_items[rndValues[t]] - 1 > 0:
                num_items[rndValues[rndValues[t]]] -= 1
                t += 1
            else:
                count += 1
                t += 1
    else:
        # addition
        count = sumupvalue - sum
        while t < count:
            num_items[rndValues[rndValues[t]]] += 1
            t += 1
    return [num_items, np.var(num_items, ddof=0)]

def checkandreturnpossiblescheduled(idx,gamma_t,threshold,computationtimet,computationthreshold):
    idxpossible = []
    for i in idx:
        if gamma_t[i] > threshold and computationtimet[i] < computationthreshold:
            idxpossible.append(i)
    return idxpossible

def initiatevariables(folder,args):
    if folder == 'FED':
        federated = 1
        rnd = 0
        pf = 0
        args.CSI = 1
        args.dataseteffect = 1
        args.R = 6
    elif folder == 'RANDOM':
        federated = 0
        rnd = 1
        pf = 0
        args.CSI = 1
        args.dataseteffect = 1
        args.R = 5
    elif folder == 'PROP':
        federated = 0
        rnd = 1
        pf = 1
        args.CSI = 1
        args.dataseteffect = 1
        args.R = 5
    elif folder == 'MY':
        federated = 0
        rnd = 0
        pf = 0
        args.CSI = 1
        args.dataseteffect = 1
        args.R = 5
    elif folder == 'YBASE':
        federated = 0
        rnd = 0
        pf = 0
        args.CSI = 1
        args.dataseteffect = 0
        args.R = 5
    elif folder == 'GPR':
        federated = 0
        rnd = 0
        pf = 0
        args.CSI = 0
        args.dataseteffect = 1
        args.R = 6
    return args, federated, rnd, pf
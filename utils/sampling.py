#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
from supp import *
import random
import numpy as np
import scipy.io as sio

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_differentdataset(dataset, num_users, a,seed):
    # np.random.seed(seed)
    # a = checkvariance(variance, num_users, len(dataset))
    numdata = len(dataset)/24
    K = num_users
    x = np.arange(1, K + 1)
    weights = np.float_power(x, (-a))
    weights /= weights.sum()
    sample = weights * numdata
    num_items = np.around(sample)
    sum = num_items.sum()
    rndValues = np.random.choice(K, 10 * K, replace=True)
    t = 0
    if sum - numdata > 0:
        # deduct
        count = sum - numdata
        while t <= count:
            if num_items[rndValues[t]] - 1 > 0:
                num_items[rndValues[rndValues[t]]] -= 1
                t += 1
            else:
                count += 1
                t += 1
    else:
        # addition
        count = numdata - sum
        while t < count:
            num_items[rndValues[rndValues[t]]] += 1
            t += 1
    print(num_items)
    variance = np.var(num_items, ddof=0)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, int(num_items[i]), replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return [dict_users, num_items, variance]


def mnist_noniid(dataset, num_users, alpha, sigma, cifar):
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(len(dataset))
    if cifar:
        labels = dataset.targets
    else:
        labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    num_class = 10

    ##
    numdata = 2500
    K = num_users
    x = np.arange(1, K + 1)
    weights = np.float_power(x, (-sigma))
    weights /= weights.sum()
    sample = weights * numdata
    num_items = np.around(sample)
    sum = num_items.sum()
    rndValues = np.random.choice(K, 10 * K, replace=True)
    t = 0
    if sum - numdata > 0:
        # deduct
        count = sum - numdata
        while t <= count:
            if num_items[rndValues[t]] - 1 > 0:
                num_items[rndValues[rndValues[t]]] -= 1
                t += 1
            else:
                count += 1
                t += 1
    else:
        # addition
        count = numdata - sum
        while t < count:
            num_items[rndValues[rndValues[t]]] += 1
            t += 1
    print(num_items)

    # put idx into class wise idx
    classviseidx = {i: np.array([], dtype='int64') for i in range(num_class)}
    for i in range(len(dataset)):
        classviseidx[idxs_labels[1, i]] = np.append(classviseidx[idxs_labels[1, i]], idxs_labels[0, i])
    output = {'user' + str(i): np.array([], dtype='int64') for i in range(num_users)}
    # Dirichelet distributed idx distribution
    if alpha > 0:
        classViseallocationoverUsers = {i: np.array([], dtype='int64') for i in range(num_users)}
        output = {'user' + str(i): np.array([], dtype='int64') for i in range(num_users)}
        if alpha > 10000000000000:
            # consider as infinity
            for i in range(num_users):
                k = np.array([], dtype='int64')
                for j in range(num_class):
                    k = np.append(k, 1/num_class)
                classViseallocationoverUsers[i] = np.round(k * num_items[i])
                # output['user' + str(i)] = np.round(k * totalperuser)
        else:
            for i in range(num_users):
                k = np.random.gamma(alpha, 1, num_class)
                k = k / np.sum(k)
                classViseallocationoverUsers[i] = np.round(k*num_items[i])
                # output['user' + str(i)] = np.round(k * totalperuser)

        # divide and assign
        for i in range(num_users):
            for j in range(num_class):
                rand_set = np.random.choice(classviseidx[j], int(classViseallocationoverUsers[i][j]), replace=False)
                # remove selected list
                templst=list(classviseidx[j])
                for rem_itm in rand_set:
                    templst.remove(rem_itm)
                classviseidx[j] = np.array(templst, dtype='int64')
                dict_users[i] = np.append(dict_users[i], rand_set)
    else:
        for i in range(num_users):
            k = np.zeros(10)
            j = random.randint(0, 9)
            k[j] = 1
            rand_set = np.random.choice(classviseidx[j], int(num_items[i]), replace=False)
            templst = list(classviseidx[j])
            for rem_itm in rand_set:
                templst.remove(rem_itm)
            classviseidx[j] = np.array(templst, dtype='int64')
            dict_users[i] = np.append(dict_users[i], rand_set)
            output['user' + str(i)] = np.round(k * num_items[i])

    # number of items and variance computation
    num_items = []
    for indexterm in dict_users:
        num_items.append(len(dict_users[indexterm]))
    num_items = np.array(num_items)
    variance = np.var(num_items, ddof=0)
    sio.savemat('classvisenoniid.mat', output)
    return [dict_users, num_items, variance]


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 50
    d = mnist_noniid(dataset_train, num)

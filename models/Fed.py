#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn

def FedAvg(w, nk, idx_scheduled):
    tot = sum(nk[idx_scheduled[0:len(w)]])
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = nk[idx_scheduled[0]]*w[0][k]
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += nk[idx_scheduled[i]]*w[i][k]
        w_avg[k] = torch.div(w_avg[k], tot)
    return w_avg

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments
    # parser.add_argument('--epochs', type=int, default=50, help="rounds of training")                # GLOBAL AVERAGING ROUNDS
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")            # NO OF WORKERS
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")    # NO OF LOCAL SGD ITR
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")             # SGD BATCH SIZE
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")                     # LEARNING RATE
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")  # MOMENTUM FOR SGD
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True', help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid',  type=int, default=0, help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

    # prediction parameters
    parser.add_argument('--datalength', type=int, default=1000000, help="size of one transaction in bits")
    parser.add_argument('--crieteriaselection', type=str, default='threshold',help='schedulling criteria')

    # scheduler
    parser.add_argument('--T', type=int, default=100, help='global-iterations')
    parser.add_argument('--R', type=int, default=10, help='resource blocks')
    parser.add_argument('--beta', type=int, default=0.7, help='beta value')
    parser.add_argument('--gammma_th', type=float, default=1, help='SINR threshold')
    parser.add_argument('--delta', type=float, default=0.5, help='SINR GPR prediction variance threshold')
    parser.add_argument('--pi', type=float, default=1, help='tradeoff parameter between queue length q_t and optimality of the solution')
    parser.add_argument('--pi1', type=float, default=1, help='tradeoff parameter between queue length without CSI')
    parser.add_argument('--pi2', type=float, default=1, help='tradeoff parameter between without CSI')
    parser.add_argument('--CSI', type=int, default=1, help='CSI available')
    args = parser.parse_args()
    return args

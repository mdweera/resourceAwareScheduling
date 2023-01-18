import numpy as np
from scheduler import Scheduler
from utils.options import args_parser
import torch
from torchvision import datasets, transforms
from utils.sampling import mnist_iid, mnist_noniid
from models.Nets import MLP, mydefonelayer, mydefonelayerCNN, mydefseverallayerCNN
import time
import copy
from models.Fed import FedAvg
from models.Update import LocalUpdate
from models.test import test_img, trainaccuracy_img
import scipy.io as sio
from supp import *
import random
# initializations
args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

# parameters (change args)
folder = 'RANDOM'
args, federated, rnd, pf = initiatevariables(folder, args)

def mainfunctionrun(folder):
    alpha = 1000000000000000
    cifar10 = 0
    computationth = 1.2
    args.num_users = 10
    args.gammma_th = 1.2
    args.T = 100
    args.free = 20
    # [1.98, 1.36, 1.0170, 0.76, 0.5450, 0.3510, 0.1710]
    sigma = 0
    args.noofpreviousvalues = 20
    args.local_ep = 8
    topo = range(100)

    # load data
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    if cifar10 == 0:
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
    else:
        transform_cifar10 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10(root='./data', train=True,download=True, transform=transform_cifar10)
        # trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
        dataset_test = datasets.CIFAR10(root='./data', train=False,download=True, transform=transform_cifar10)
        # testloader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=False, num_workers=2)
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    for topology in topo:
        for args.num_users in [10]:
            np.random.seed(topology)
            random.seed(topology)
            scheculer = Scheduler(args)
            # add free five data set obtained from measuring
            scheculer.knownCSIforpredict[:, :, 0:(args.free)] = scheculer.channels[:, :, 0:(args.free)]
            print('@@@@@@@@@@@@@@@ for users {:} topo {:} @@@@@@@@@@@@@@@@@@@@'.format(args.num_users, topology))
            if args.num_users > 1:
                dict_users, n_k, variance = mnist_noniid(dataset_train, args.num_users, alpha, sigma, cifar10)
            else:
                # CENTRALIZED TRAINING
                dict_users, n_k, variance = mnist_noniid(dataset_train, 10, 10000000000000000000, 0, cifar10)
                temp_dict_users = []
                for t1 in range(10):
                    for t2 in range(len(dict_users[t1])):
                        temp_dict_users.append(dict_users[t1][t2])
                dict_users[0] = temp_dict_users
                n_k = len(dict_users[0])
            img_size = dataset_train[0][0].shape
            len_in = 1
            for x in img_size:
                len_in *= x
            print('Data loaded {:03d} samples and starting building model'.format(len(dataset_train)))
            # timing
            start = time.process_time()
            milestoneiter = []
            # build model
            if cifar10 != 1:
                net_glob = mydefonelayer(dim_in=len_in, dim_out=args.num_classes).to(args.device)
            else:
                net_glob = mydefseverallayerCNN(dim_out=args.num_classes).to(args.device)
            print('model build complete start training...')

            loss_train, accuracy_train, global_averagedweights, idx_scheduled = [], [], [], []
            inferenceaccuracy,inferenceloss = [], []
            timeconsumed = []
            feasibleidx = [] # idx that are possible to schedule due to chennels
            scheduledidx_random = [] # idx that are scheduled randomly
            # scheduling variables
            print('start training scheduling...')
            if federated == 0:
                q = [0.1]
                auxilary = [0]
                nu = [0]
                ut = []

            for t in range(args.T):
                # IN THE ITERATIONS
                w_locals_updated = []
                milestoneiter.append(time.process_time())
                print('Start of iteration {:2d} :  time consumed : {:3f}'.format(t, milestoneiter[t] - start if t == 0 else
                milestoneiter[t] - milestoneiter[t - 1]))
                net_glob.train()

                # SCHEDULING
                idxusers_1 = []
                if args.num_users > 1:
                    if federated == 1:
                        idx_scheduled = np.random.choice(args.num_users, args.num_users, replace=False)
                    elif federated == 0:
                        if rnd == 0:
                            if args.CSI == 1:
                                idx_scheduled, q, auxilary, ut = scheculer.scheduleCSI(t, args.T, n_k, q, args.R, args.num_users,args.gammma_th, args.pi, nu, ut, args.beta, args.dataseteffect, computationth)
                            else:
                                idx_scheduled, q, auxilary, ut = scheculer.scheduleWITHOUTCSI(t, args.T, n_k, q, args.R, args.num_users,args.gammma_th, args.pi1, args.pi2, nu, ut, args.beta, args.free, args.noofpreviousvalues,computationth)
                        else:
                            if pf ==0:
                                np.random.seed(t)
                                idxusers_1 = np.random.choice(args.num_users, args.R, replace=False)
                                idx_scheduled = checkandreturnpossiblescheduled(idxusers_1, scheculer.channels[:, 0, t], args.gammma_th, scheculer.computationtime[:, t], computationth)
                            else:
                                #pf=1
                                np.random.seed(t)
                                idxusers_1 = scheculer.schedulePF(t, args.T, args.R, args.num_users)
                                idx_scheduled = checkandreturnpossiblescheduled(idxusers_1, scheculer.channels[:, 0, t], args.gammma_th,
                                                                                scheculer.computationtime[:, t], computationth)
                                for itr in range(len(idx_scheduled)):
                                    scheculer.scheduledfreq[idx_scheduled[itr]] += 1
                else:
                    idx_scheduled = [0]  # for centralized learning
                str1 = ''.join(str('{:}'.format(e)) + ', ' for e in idx_scheduled)
                print('['+str1+']'+'{:} idx  scheduled for iteration {:} :  time consumed : {:3f}'.format(len(idx_scheduled), t, time.process_time() - milestoneiter[t]))

                # LOCAL UPDATING
                iterationuser = 0
                trsttime = time.process_time()
                scheduledidx_random.append(idxusers_1)
                feasibleidx.append(idx_scheduled)
                if len(idx_scheduled) > 0:
                    for useridx in range(args.num_users):
                        if useridx in idx_scheduled:
                            local_model = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[useridx])
                            w, loss = local_model.train(net=copy.deepcopy(net_glob).to(args.device))
                            w_locals_updated.append(copy.deepcopy(w))
                        if iterationuser % 10 == 0:
                            print('{:3d} of {:3d} , '.format(iterationuser, args.num_users, end="\n"))
                        iterationuser += 1
                    if args.num_users > 1:
                        w_glob = FedAvg(w_locals_updated, n_k, idx_scheduled)
                    else:
                        w_glob = w_locals_updated[0]
                    net_glob.load_state_dict(w_glob)
                    # testing per epoch
                    net_glob.eval()
                    acc_train, loss_training = trainaccuracy_img(net_glob, dataset_train, args, dict_users)
                    print('Round{:} RESULTS: Accuracy is :{:}% Training loss loss {:.3f} consumed time for training {:.3f}'.format(t, acc_train, loss_training, time.process_time() - trsttime), end="\n \n")
                    if t == (args.T - 1) or t == 10 or t == 50 or t == 75 or t == 0 or t == 1:
                        acc_train_t, loss_training_t = test_img(net_glob, dataset_test, args)
                        print('Round{:} RESULTS: Accuracy is :{:}% Training loss loss {:.3f} consumed time for training {:.3f}'.format(t, acc_train_t, loss_training_t, time.process_time() - trsttime), end="\n \n")
                        inferenceaccuracy.append(acc_train_t)
                        inferenceloss.append(loss_training_t)
                    timeconsumed.append(time.process_time() - trsttime)
                else:
                    acc_train = 0
                    loss_training = 0
                    print('Round{:} RESULTS: Accuracy is :{:}% Training loss loss {:.3f} consumed time for training {:.3f}'.format(t, acc_train, loss_training, time.process_time() - trsttime), end="\n \n")
                accuracy_train.append(acc_train)
                loss_train.append(loss_training)

                # OUTPUTS WRITE FILES
                if t == (args.T - 1) or t == 25 or t == 50 or t == 75 or t == 1 or t == 5:
                    a = {}
                    a['scheduleduderid'] = scheculer.scheduledidx
                    a['channelsoriginal'] = scheculer.channelscheduler
                    a['lossofiterations'] = loss_train
                    a['accuracy'] = accuracy_train
                    a['inferenceaccuracy'] = inferenceaccuracy
                    a['inference_f'] = inferenceloss
                    a['feasibleidx'] = feasibleidx
                    a['scheduledidx_random'] = scheduledidx_random
                    if federated == 0:
                        a['queue'] = q
                        a['auxilary'] = auxilary
                        a['ut'] = ut
                    a['timeconsumedforcomputeiteration'] = timeconsumed
                    a['Infovarianceactualvarpi1pi2betaT'] = [ variance, args.pi1, args.pi2, args.beta, args.T]
                    a['datapointnumbers'] = n_k
                    str1 = 'Res/'+folder+'/USERS_'+str(args.num_users)+'_topo_'+str(topology)+'_pi1_'+str(args.pi1)+'pi2_'+str(args.pi2)+'uptoT_'+str(args.T)
                    if args.CSI == 1:
                        sio.savemat(str1+'_CSI.mat', a)
                    else:
                        a['predictedchannels'] = scheculer.channelpredictions
                        sio.savemat(str1+'_withoutCSI.mat', a)
                if t == (args.T - 1) or t == 1:
                    # print(str1)
                    torch.save(net_glob.state_dict(), 'Model/'+str1+'.pwf')
                    aux = {}
                    tempVar = []
                    str2 = folder+'/'+'topo_'+str(topology)
                    for useridx in range(args.num_users):
                        acc_train, loss_training = trainaccuracy_img(net_glob, dataset_train, args, [dict_users[useridx]])
                        tempVar.append([useridx, len(dict_users[useridx]), acc_train, loss_training])
                    aux['userAccuracy'] = tempVar
                    sio.savemat('peruseraccuracy/' + str2 + '.mat', aux)


mainfunctionrun('MY')

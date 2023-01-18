import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, RBF
import scipy.io as sio
from pulp import *
from supp import *
np.random.seed(1)  # random numbers predictable


def retrievechannels(args):
    num_users = args.num_users
    T = args.T
    R = args.R
    power = 10
    n_0 = 10
    totT = T + args.free
    availability = 1
    try:
        mat_contents = sio.loadmat('channelcorrelated.mat')
    except:
        availability = 0

    if availability == 0:
        channels = np.zeros((num_users, R, T + args.free))
        a = {}
        for i in range(T + args.free):
            for k in range(num_users):
                for r in range(R):
                    sq_channel_abs = np.random.rayleigh()
                    channels[k, r, i] = power * sq_channel_abs / n_0
        a['data'] = channels
        a['power'] = power
        a['n_0'] = n_0
        sio.savemat('data', a)
    else:
        channels_temp = mat_contents['data']
        channels = channels_temp[0:num_users, 0:R, 0:totT]
    return channels

def retrievecomputationtime():
    computationtime_tmp = sio.loadmat('computationtime.mat')
    computationtime = computationtime_tmp['computationtime']
    return computationtime

def predict(knownCSIforpredict, K, R, t, free, noofpreviousvalues,gamma_t2):
    gamma = np.zeros([K, R])
    variance = np.zeros([K, R])
    gp_kernel = ExpSineSquared(1.0, 5.0)
    gp = GaussianProcessRegressor(kernel=gp_kernel, n_restarts_optimizer=9)
    errorcount = 0
    for k in range(K):
        for r in range(R):
            gamma_tmp = []
            for i in range(t+free):
                gamma_tmp.append(knownCSIforpredict[k][r][i])
            if not sys.warnoptions:
                warnings.simplefilter("ignore")
                os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses
            # check whether this user have past data
            if len(np.nonzero(gamma_tmp)[0]) != 0:
                a = np.atleast_2d(gamma_tmp).T
                indexes = np.nonzero(a)[0]
                if len(indexes) > noofpreviousvalues:
                    indexes = indexes[len(indexes)-noofpreviousvalues:len(indexes)]
                gp.fit(np.atleast_2d(indexes).T, a[indexes])
                y_pred, sigma = gp.predict(np.atleast_2d(np.linspace(0, t+free, t+free+1)).T, return_std=True)
                gamma[k][r] = y_pred[t+free]
                if np.absolute(gamma_t2[k][r] - gamma[k][r]) > 1:
                    errorcount += 1
                variance[k][r] = sigma[t+free]
                if k % 5 == 0 and r == 0:
                    str1 = ''.join(str('{:.3f}'.format(e)) + ', ' for e in gamma_tmp)
                    print(('channel 1 for user ' + str(k) + ' are [' + str1 + '] and predicted to be {:.3f} with variance {:.3f}').format(y_pred[t+free][0], sigma[t+free]))
    print('channel error count:'+str(errorcount)+'out of :'+str(K*R))
    return [gamma, variance]


class Scheduler:
    def __init__(self, args):
        self.channels = retrievechannels(args)
        self.computationtime =retrievecomputationtime()
        self.scheduledidx = []
        self.scheduledfreq = np.ones(args.num_users)
        self.channelscheduler = []
        self.channelpredictions = []
        self.knownCSIforpredict = np.zeros([args.num_users, args.R, args.T + args.free])

    def scheduleCSI(self, t, T, dataset, q, R, K, gamma_th, pi, nu, ut, beta, effectofdataset,computationth):
        print('============ scheduling for iter {:2d} ============='.format(t))
        gamma_t = self.channels[:, :, t]
        computationtime_t = self.computationtime[:, t]
        no_users = K
        summation = 0
        for i in range(len(dataset)):
            summation = summation + dataset[i]
        n = summation

        # create
        prob = LpProblem("problem", LpMaximize)

        # scheduling variable
        schedule = []
        for k in range(K):
            schedule.append(LpVariable('s{}'.format(k), lowBound=0., upBound=1, cat='Continuous'))

        # resource variables
        resource = []
        for k in range(K):
            for r in range(R):
                resource.append(LpVariable('lambda{}_{}'.format(k, r), lowBound=0., upBound=1, cat='Continuous'))

        # objective
        if effectofdataset == 1:
            firstsummation = 0
            for k in range(no_users):
                firstsummation = firstsummation + q[t] * (1 - beta) * dataset[k] / n * schedule[k]
            prob += firstsummation
        else:
            firstsummation = 0
            for k in range(no_users):
                firstsummation = firstsummation + q[t] * (1 - beta) * schedule[k]
            prob += firstsummation

        # constraints
        sumation1 = 0
        for k in range(K):
            sumation1 = sumation1 + schedule[k]
        prob += sumation1 <= R

        for k in range(K):
            summation2 = 0
            for r in range(R):
                summation2 = summation2 + resource[k * R + r]
            prob += schedule[k] <= summation2

        for k in range(K):
            for r in range(R):
                indicator1 = 0
                if gamma_t[k][r] > gamma_th:
                    indicator1 = 1
                prob += resource[k * R + r] <= indicator1

        for k in range(K):
            indicator = 0
            if computationtime_t[k] < computationth:
                indicator = 1
            prob += schedule[k] <= indicator

        for r in range(R):
            summation3 = 0
            for k in range(K):
                summation3 = summation3 + resource[k * R + r]
            prob += summation3 <= 1

        for k in range(K):
            summation5 = 0
            for r in range(R):
                summation5 = summation5 + resource[k * R + r]
            prob += summation5 <= 1

        # Solve
        prob.solve()
        print("Status:", LpStatus[prob.status])

        idxusers = []
        summation = 0
        for k in range(K):
            if schedule[k].varValue > .5:
                idxusers.append(k)
            summation = summation + dataset[k]*schedule[k].varValue/n

        ut.append((1 - beta) * summation)

        print(str(len(idxusers)) + 'are scheduled')

        channelsscheduled = []
        for k in range(K):
            userresource = []
            for r in range(R):
                userresource.append(resource[k*R+r].varValue)
            channelsscheduled.append(userresource)         # {user : [channel schedule list]}

        for k in range(K):
            if k in idxusers:
                print(channelsscheduled[k])

        # solving for nu
        if (q[t]+pi*T*(1-np.average(nu))**(T-1))*n > 0:
            nu.append(1)
        else:
            nu.append(0)

        if q[t] + nu[t] - ut[t] > 0:
            q.append(q[t] + nu[t] - ut[t])
        else:
            q.append(0)
        self.scheduledidx.append(idxusers)
        self.channelscheduler.append(channelsscheduled)
        return [idxusers, q, nu, ut]

    def scheduleWITHOUTCSI(self, t, T, dataset, q, R, K, gamma_th, pi_1, pi_2, nu, ut, beta, free, noofpreviousvalues, computationth):
        print('============ scheduling for iter {:2d} ============='.format(t))
        gamma_t2 = self.channels[:, :, t]
        gamma_t, variance = predict(self.knownCSIforpredict, K, R, t, free, noofpreviousvalues,gamma_t2)
        self.channelpredictions.append([gamma_t])          # {iteration : predictions}
        computationtime_t = self.computationtime[:, t]
        no_users = K
        summation = 0
        for i in range(len(dataset)):
            summation = summation + dataset[i]
        n = summation

        # create
        prob = LpProblem("problem", LpMaximize)

        # scheduling variable
        schedule = []
        for k in range(K):
            schedule.append(LpVariable('s{}'.format(k), lowBound=0., upBound=1, cat='Continuous'))

        # resource variables
        resource = []
        for k in range(K):
            for r in range(R):
                resource.append(LpVariable('lambda{}_{}'.format(k, r), lowBound=0., upBound=1, cat='Continuous'))

        # objective
        firstsummation = 0
        for k in range(no_users):
            firstsummation = firstsummation + q[t] * (1 - beta) * dataset[k] / n * schedule[k]
        sum2 = 0
        for k in range(no_users):
            for r in range(R):
                sum2 += pi_2*(1-resource[k*R+r])*variance[k, r]
        prob += firstsummation - sum2

        # constraints
        sumation1 = 0
        for k in range(K):
            sumation1 = sumation1 + schedule[k]
        prob += sumation1 <= R

        for k in range(K):
            summation2 = 0
            for r in range(R):
                summation2 = summation2 + resource[k * R + r]
            prob += schedule[k] <= summation2

        for k in range(K):
            for r in range(R):
                I = 0
                if gamma_t[k][r] > gamma_th:
                    I = 1
                prob += resource[k * R + r] <= I

        for k in range(K):
            indicator = 0
            if computationtime_t[k] < computationth:
                indicator = 1
            prob += schedule[k] <= indicator

        for r in range(R):
            summation3 = 0
            for k in range(K):
                summation3 = summation3 + resource[k * R + r]
            prob += summation3 <= 1

        for k in range(K):
            summation5 = 0
            for r in range(R):
                summation5 = summation5 + resource[k * R + r]
            prob += summation5 <= 1

        # Solve
        prob.solve()
        print("Status:", LpStatus[prob.status])

        idxusers = []
        summation = 0
        for k in range(K):
            if schedule[k].varValue > .5:
                idxusers.append(k)
            summation = summation + dataset[k] * schedule[k].varValue / n
        ut.append((1 - beta) * summation)
        self.scheduledidx.append(idxusers)
        idxusers = []
        channelsscheduled = {}
        for k in range(K):
            userresource = []
            for r in range(R):
                userresource.append(resource[k * R + r].varValue)
                if resource[k * R + r].varValue > 0.5 and gamma_t2[k][r] > gamma_th -1:
                    idxusers.append(k)
            channelsscheduled.update({k: userresource})  # user : [channel schedule list]

        print(str(len(idxusers)) + 'are scheduled')
        for k in range(K):
            if k in idxusers:
                print(channelsscheduled[k])
        # solving for nu
        if (q[t] + pi_1 * T * (1 - np.average(nu)) ** (T - 1)) * n > 0:
            nu.append(1)
        else:
            nu.append(0)

        if q[t] + nu[t] - ut[t] > 0:
            q.append(q[t] + nu[t] - ut[t])
        else:
            q.append(0)
        self.channelscheduler.append(channelsscheduled)
        for k in range(K):
            for r in range(R):
                if k in idxusers:
                    #if channelsscheduled[k][r] > 0.5:
                    self.knownCSIforpredict[k, r, t + free] = self.channels[k, r, t]
        return [idxusers, q, nu, ut]

    def schedulePF(self, t, T, Resourceblocks, num_users):
        PF_coeff = 1/self.scheduledfreq
        idx_scheduled = []
        print(self.scheduledfreq)
        for r in range(Resourceblocks):
            index_r_arr = np.where(PF_coeff == max(PF_coeff))
            index_r = index_r_arr[0][0]
            idx_scheduled.append(index_r)
            PF_coeff[index_r] = 0
        return idx_scheduled













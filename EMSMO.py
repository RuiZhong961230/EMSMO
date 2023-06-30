import os
import numpy as np
from opfunu.cec_based import cec2013
import math
from pyDOE2 import lhs


PopSize = 100                                                  # the number of individuals (PopSize > 4)
DimSize = 10                                                    # the number of variables
LB = -100                                                 # the maximum value of the variable range
UB = 100                                                  # the minimum value of the variable range
trials = 30                                                   # the number of independent runs
MaxFEs = DimSize * 500                      # the maximum number of fitness evaluations
MaxIter = int(MaxFEs / PopSize)

Population = np.zeros((PopSize, DimSize))
Population_fitness = np.zeros(PopSize)

Offspring = np.zeros((PopSize, DimSize))
Offspring_fitness = np.zeros(PopSize)
varepsilon = 0.00000001
Fun_num = 1

ScoreMatrix = np.zeros((4, MaxIter))


def SMclear():
    global ScoreMatrix
    for i in range(len(ScoreMatrix)):
        for j in range(len(ScoreMatrix[i])):
            ScoreMatrix[i][j] = varepsilon


def scales(data):
    data = np.array(data)
    limit_scale = []
    for i in range(len(data[0])):
        d = data[:, i]
        limit_scale.append([min(d), max(d)])
    return limit_scale



def CheckIndi(Indi):
    range_width = UB - LB
    for i in range(DimSize):
        if Indi[i] > UB:
            n = int((Indi[i] - UB) / range_width)
            mirrorRange = (Indi[i] - UB) - (n * range_width)
            Indi[i] = UB - mirrorRange
        elif Indi[i] < LB:
            n = int((LB - Indi[i]) / range_width)
            mirrorRange = (LB - Indi[i]) - (n * range_width)
            Indi[i] = LB + mirrorRange
        else:
            pass


def Initialization(Func):
    global Population, Population_fitness, DimSize, PopSize
    for i in range(PopSize):
        for j in range(DimSize):
            Population[i][j] = np.random.uniform(LB, UB)
        Population_fitness[i] = Func(Population[i])


def normal(Fits):
    w = np.zeros(len(Fits))
    sums = sum(Fits)
    for i in range(len(Fits)):
        w[i] = Fits[i] / sums
    return w


def Search(Func, i, t):
    global Population, Population_fitness, Offspring, Offspring_fitness
    x_best = Population[np.argmin(Population_fitness)]

    r1, r2 = np.random.randint(0, PopSize), np.random.randint(0, PopSize)
    while r1 == r2:
        r2 = np.random.randint(0, PopSize)
    vc = 1 - (t / MaxIter)
    a = math.atanh(vc)
    if np.random.rand() < 0.5:
        Offspring[i] = Population[i] + (np.random.uniform(-a, a) + 0.1) * (x_best - Population[i]) + (
                    np.random.uniform(-a, a) + 0.1) * (Population[r1] - Population[r2])
    else:
        Offspring[i] = x_best + (np.random.uniform(-a, a) + 0.1) * (Population[r1] - Population[r2])

    CheckIndi(Offspring[i])
    Offspring_fitness[i] = Func(Offspring[i])


def Approach(Func, i, t):
    global Population, Population_fitness, Offspring, Offspring_fitness
    X_mean = np.zeros(DimSize)
    idx = np.argsort(Population_fitness)
    size = np.random.randint(1, int(PopSize / 2))
    for j in range(size):
        X_mean += Population[idx[j]]
    X_mean /= size
    for j in range(DimSize):
        Offspring[i][j] = X_mean[j] + 4 * np.random.rand() - 2
    CheckIndi(Offspring[i])
    Offspring_fitness[i] = Func(Offspring[i])


def weights(Fits):
    w = np.zeros(len(Fits))
    sums = 0
    for i in range(len(Fits)):
        sums += 1 / Fits[i]
    for i in range(len(Fits)):
        w[i] = (1 / Fits[i]) / sums
    return w


def Wrap(Func, i, t, k=10):
    global Population, Population_fitness, Offspring, Offspring_fitness
    samples = list(range(PopSize))
    samples.remove(i)
    samples = np.random.permutation(samples)[:k]
    subPop = Population[samples]
    subFit = Population_fitness[samples]

    W = weights(subFit)
    Off = np.zeros(DimSize)
    for j in range(len(W)):
        Off += W[j] * subPop[j]
    Offspring[i] = Off

    CheckIndi(Offspring[i])
    Offspring_fitness[i] = Func(Offspring[i])


def Mutate(Func, i, t):
    global Offspring, Offspring_fitness
    for j in range(DimSize):
        Offspring[i][j] = np.random.uniform(LB, UB)
    Offspring_fitness[i] = Func(Offspring[i])


def Select():
    global Population, Population_fitness, Offspring, Offspring_fitness
    for i in range(PopSize):
        if Offspring_fitness[i] < Population_fitness[i]:
            Population[i] = Offspring[i]
            Population_fitness[i] = Offspring_fitness[i]


def SequenceConstruct(t, size):
    global ScoreMatrix, Fun_num, DimSize
    sequence = np.zeros(PopSize)
    subscore = np.zeros(size)
    for i in range(size):
        for j in range(t):
            subscore[i] += ScoreMatrix[i][j]
    w = normal(subscore)
    for i in range(1, len(w)):
        w[i] += w[i-1]

    for i in range(PopSize):
        r = np.random.uniform(0, 1)
        for j in range(len(w)):
            if r < w[j]:
                sequence[i] = j
                break
    return sequence


def ScoreUpdate(sequence, size, t):
    global Population, Population_fitness, Offspring, Offspring_fitness, ScoreMatrix
    times = np.zeros(size)
    improve_time = np.zeros(size)
    improve_amount = np.zeros(size)
    base = varepsilon
    for i in range(len(sequence)):
        base += max((Population_fitness[i] - Offspring_fitness[i]), 0)
    for s in sequence:
        times[int(s)] += 1
    for i in range(len(sequence)):
        if Offspring_fitness[i] < Population_fitness[i]:
            improve_time[int(sequence[i])] += 1
            improve_amount[int(sequence[i])] += max((Population_fitness[i] - Offspring_fitness[i]), 0) / base
    for i in range(size):
        improve_time[i] = improve_time[i] / (times[i] + varepsilon)
    sum_t = sum(improve_time)
    for i in range(size):
        improve_time[i] /= (sum_t + varepsilon)
    for i in range(size):
        ScoreMatrix[i][t] = 0.5 * improve_time[i] + 0.5 * improve_amount[i]
    return


def EMSMO(Func, t):
    global Population, Population_fitness, Offspring, Offspring_fitness, ScoreMatrix
    archive = [Search, Approach, Wrap, Mutate]
    size = len(archive)
    sequence = SequenceConstruct(t, size)

    for i in range(PopSize):
        archive[int(sequence[i])](Func, i, t)
    ScoreUpdate(sequence, size, t)
    Select()


def RunEMSMO(Func):
    global MaxIter, Fun_num, Population_fitness, ScoreMatrix
    All_Trial_Best = []
    for i in range(trials):                 # run the algorithm independently multiple times
        Best_list = []
        iteration = 1
        np.random.seed(2022 + 88*i)                 # fix the seed of random number
        Initialization(Func)                            # randomly initialize the population
        Best_list.append(min(Population_fitness))
        SMclear()
        while iteration <= MaxIter:
            EMSMO(Func, iteration)
            iteration += 1
            Best_list.append(min(min(Population_fitness), Best_list[-1]))
        All_Trial_Best.append(Best_list)
    np.savetxt('./EMSMO_Data/CEC2013/F{}_{}D.csv'.format(Fun_num, DimSize), All_Trial_Best, delimiter=",")


def main(Dim):
    global Fun_num, DimSize, MaxFEs, MaxIter, Population, Offspring, ScoreMatrix
    DimSize = Dim
    MaxFEs = DimSize * 500
    MaxIter = int(MaxFEs / PopSize)
    ScoreMatrix = np.zeros((4, MaxIter+1))
    Population = np.zeros((PopSize, DimSize))
    Offspring = np.zeros((PopSize, DimSize))

    CEC2013Funcs = [cec2013.F12013(Dim), cec2013.F22013(Dim), cec2013.F32013(Dim), cec2013.F42013(Dim),
                    cec2013.F52013(Dim), cec2013.F62013(Dim), cec2013.F72013(Dim), cec2013.F82013(Dim),
                    cec2013.F92013(Dim), cec2013.F102013(Dim), cec2013.F112013(Dim), cec2013.F122013(Dim),
                    cec2013.F132013(Dim), cec2013.F142013(Dim), cec2013.F152013(Dim), cec2013.F162013(Dim),
                    cec2013.F172013(Dim), cec2013.F182013(Dim), cec2013.F192013(Dim), cec2013.F202013(Dim),
                    cec2013.F212013(Dim), cec2013.F222013(Dim), cec2013.F232013(Dim), cec2013.F242013(Dim),
                    cec2013.F252013(Dim), cec2013.F262013(Dim), cec2013.F272013(Dim), cec2013.F282013(Dim)]

    for i in range(20, len(CEC2013Funcs)):
        Fun_num = i + 1
        RunEMSMO(CEC2013Funcs[i].evaluate)


if __name__ == "__main__":
    if os.path.exists('./EMSMO_Data/CEC2013') == False:
        os.makedirs('./EMSMO_Data/CEC2013')
    Dims = [10, 30, 50]
    for Dim in Dims:
        main(Dim)

import numpy as np
import numpy.random as rnd
import random
import math

from BasicFunctions import distances, sum_way, answer_loop
from SimulatedAnnealing import sa_main


def find_next_town(probability):
    coin = random.random()
    # probability_transform = abs(probability - coin)
    probability_transform = probability
    p = probability_transform[0]
    index = 0
    for i in range(len(probability_transform)):
        # if probability_transform[i] < p:
        if probability_transform[i] > p:
            p = probability_transform[i]
            index = i
    return index


def ac_main():
    # инициализация параметров алгоритма
    name, dim, coords, distance = distances()
    q = 5
    alfa = 3
    beta = 1
    p = 0.8
    tmax = 10

    # инициализация ребер
    eta = np.power(distance, -1, dtype=np.float64)
    tau = np.ones((dim, dim), dtype=np.float64)

    # размещение муравьев в случайно выбранные вершины
    multiplier = 2
    num_ant = multiplier * dim  # - число муравьев
    # tops = np.arange(0, dim)  # набор вершин
    tops = [x for x in range(dim)]

    J = []
    P = np.zeros((num_ant, dim, dim), dtype=np.float64)
    Tk = np.zeros((num_ant, dim), dtype=np.int8)

    top_ants = rnd.permutation(dim)
    for i in range(multiplier - 1):
        top_ants = np.concatenate((top_ants, rnd.permutation(dim)))  # размещение муравьев по вершинам

    # Выбор начального кратчайшего маршрута и определение L*
    tbest, lbest = sa_main()

    print('sa', tbest, lbest)
    tbest = tbest[0: -1]

    # tbest = np.arange(0, dim)
    # lbest = sum_way(tbest, distance)

    Tk[0] = tbest
    J.append([0 for i in range(dim)])
    for ant in range(1, num_ant):
        J.append([])
        for top in tops:
            J[ant].append(top)

    Lk = np.zeros(num_ant)
    Lk[0] = lbest

    # цикл по времени жизни колонии
    for t in range(tmax):
        sum_delta_tau = np.zeros((dim, dim))
        for ant in range(1, num_ant):
            for i, top in enumerate(tops):
                J[ant][i] = top
            Tk[ant] = np.zeros(dim, dtype=np.int8)
            index = J[ant].index(top_ants[ant])
            J[ant][index] = -1
            Tk[ant][0] = index

            sum_tabu_list = 0
            # Построить маршрут Тк(t) на основе распределения вероятности по ребрам и расчитать Lk
            delta_tau = np.zeros((num_ant, dim, dim), dtype=np.float64)

            # Просчитаем delta_tau
            for i in range(len(Tk[ant]) - 1):
                delta_tau[ant-1][Tk[ant-1][i]][Tk[ant-1][i + 1]] = q / Lk[ant-1]

            sum_delta_tau += delta_tau[ant-1]

            # Просчитаем теперь tau, вероятности перехода и построим маршрут
            for i in range(dim):
                for j in range(dim):
                    tau[i][j] = (1 - p) * tau[i][j] + sum_delta_tau[i][j]

            Tk_lenght = 1
            while Tk_lenght < dim:
                i = Tk[ant][Tk_lenght - 1]
                tabu_list = [x for x in J[ant] if x >= 0]  # filter(lambda x: x >= 0, J[ant])
                for j in range(dim):
                    is_contain = j in tabu_list
                    for l in tabu_list:
                        if math.isnan(tau[i][l]**alfa * eta[i][l]**beta) | math.isinf(tau[i][l]**alfa * eta[i][l]**beta):
                            sum_tabu_list += 0
                        else:
                            sum_tabu_list += tau[i][l] ** alfa * eta[i][l] ** beta
                    if is_contain:
                        P[ant][i][j] = (tau[i][j]**alfa * eta[i][j]**beta)/sum_tabu_list
                    else:
                        P[ant][i][j] = 0
                P[ant][i][i] = 0
                check = np.sum(P[ant][i])

                Tk[ant][Tk_lenght] = find_next_town(P[ant][i] / check)
                J[ant][Tk[ant][Tk_lenght]] = -1
                Tk_lenght += 1

            Lk[ant] = sum_way(Tk[ant], distance)
            if Lk[ant] < lbest:
                lbest = Lk[ant]
                tbest = Tk[ant]
    tbest = answer_loop(tbest)
    return tbest, lbest


# result = ac_main()
# print('aco', result[0], result[1])

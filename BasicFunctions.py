from math import sqrt
import numpy as np


def distances():
    f = open('task.txt', 'r')
    start_coords = 0
    coordinates = []
    num = 0
    nametask = ""
    dimension = 0
    for line in f:
        line = line.strip()
        if line.find("NAME :") >= 0:
            line1 = line.split(' ')
            nametask = line1[-1]
        elif line.find("DIMENSION :") >= 0:
            line1 = line.split(' ')
            dimension = int(line1[-1])
        elif line.find("NODE_COORD_SECTION") >= 0:
            start_coords = 1
        elif start_coords == 1:
            if line.find("EOF") >= 0:
                start_coords = 0
            else:
                line = line.split(' ')
                coordinates.append([])
                for i in range(3):
                    coordinates[num].append(float(line[i]))
                num += 1

    distance = []
    for i in range(dimension):
        distance.append([])
        for j in range(dimension):
            distance[i].append(sqrt((coordinates[i][1] - coordinates[j][1])**2 + (coordinates[i][2] - coordinates[j][2])**2))
    return nametask, dimension, coordinates, distance


def distances_dist():
    f = open('task.txt', 'r')
    start_coords = 0
    distance = []
    coordinates = []
    num = 0
    nametask = ""
    dimension = 0
    for line in f:
        line = line.strip()
        if line.find("NAME :") >= 0:
            line1 = line.split(' ')
            nametask = line1[-1]
        elif line.find("DIMENSION :") >= 0:
            line1 = line.split(' ')
            dimension = int(line1[-1])
        elif line.find("NODE_COORD_SECTION") >= 0:
            start_coords = 1
        elif start_coords == 1:
            if line.find("EOF") >= 0:
                start_coords = 0
            else:
                line = line.split(' ')
                distance.append([])
                for i in range(dimension):
                    distance[num].append(float(line[i]))
                num += 1
    return nametask, dimension, coordinates, distance


def distances_cvrp():
    f = open('task1.txt', 'r')
    start_coords = 0
    start_demands = 0
    coordinates = []
    demands = []
    num = 0
    nametask = ""
    dimension = 0
    capacity = 0

    for line in f:
        line = line.strip()
        if line.find("NAME :") >= 0:
            line1 = line.split(' ')
            nametask = line1[-1]
        elif line.find("DIMENSION :") >= 0:
            line1 = line.split(' ')
            dimension = int(line1[-1])
        elif line.find("CAPACITY :") >= 0:
            line1 = line.split(' ')
            capacity = int(line1[-1])
        elif line.find("NODE_COORD_SECTION") >= 0:
            start_coords = 1
        elif start_coords == 1:
            if line.find("DEMAND_SECTION") >= 0:
                start_coords = 0
                start_demands = 1
            else:
                line = line.split(' ')
                coordinates.append([])
                for i in range(1, 3):
                    coordinates[num].append(float(line[i]))
                num += 1
        elif start_demands == 1:
            if line.find("DEPOT_SECTION") >= 0:
                start_demands = 0
            else:
                line = line.split(' ')
                demands.append(float(line[1]))

    distance = []
    for i in range(dimension):
        distance.append([])
        for j in range(dimension):
            distance[i].append(
                sqrt((coordinates[i][0] - coordinates[j][0]) ** 2 + (coordinates[i][1] - coordinates[j][1]) ** 2))
    return nametask, dimension, coordinates, distance, capacity, demands


def sum_way(x, dist):
    n = len(x)
    x = np.append(x, x[0])
    result = 0
    for i in range(n):
        result += dist[x[i]][x[i+1]]
    return result


def shift(lst):
    lst = np.append(lst, lst[0])
    lst = np.delete(lst, 0)
    return lst


def answer_loop(state):
    while state[0] != 0:
        state = shift(state)
    state = np.append(state, state[0])
    return state


def calcValue(x, dis, penalty):
    n = len(x)
    energy = 0
    for i in range(n-1):
        energy += dis[x[i]][x[i+1]]
    energy += dis[x[n-1]][x[0]]
    pen = calc_penalty(x)
    return energy + penalty * pen, penalty * pen


def calc_penalty(x):
    pen = [0 for _ in range(len(x))]
    for i in range(len(x)):
        pen[x[i]] += 1
    pen = [abs(pen[i] - 1) for i in range(len(x))]
    pen = sum(pen)
    return pen

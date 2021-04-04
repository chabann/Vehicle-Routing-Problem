from random import shuffle
from math import sqrt

import numpy as np
import time
import random
import copy

from BasicFunctions import distances_cvrp


def vectorDifference(vector_a, vector_b):
    ss = []
    if len(vector_a) > 0:
        for i in range(dimension-1):
            j = vector_b.index(vector_a[i])
            if i != j:
                for k in reversed(range(i, j)):
                    ss.append(k)
                el = vector_b[j]
                vector_b.pop(j)
                vector_b.insert(i, el)
    return ss


def velocityChange(al, vl, bl, gt):
    vlt = []
    blal = vectorDifference(bl, al)
    gal = vectorDifference(gt, al)
    if random.random() < k1:
        for so in vl:
            vlt.append(so)

    if random.random() < k2:
        for so in blal:
            vlt.append(so)

    if random.random() < k3:
        for so in gal:
            vlt.append(so)
    return vlt


def particleMove(al, vl, bl, gt):
    vl = velocityChange(al, vl, bl, gt)
    for so in vl:
        if so >= dimension - 2:
            so -= (dimension - 2)
        temp_val = al[so]
        al[so] = al[so + 1]
        al[so + 1] = temp_val
    return al, vl


def RouteFullness(x):
    cap = 0
    n = len(x)
    for i in range(n):
        cap += demands[x[i]]
    return cap <= capacity


def RouteDistance(x):
    length = 0
    x1 = copy.deepcopy(x)
    x1.append(0)
    n = len(x1)
    for i in range(n - 1):
        length += distance[x1[i]][x1[i+1]]
    length += distance[x1[n-1]][x1[0]]
    return length


def InterLocalSearch(vectors, numChoose, numTry, vector_lens):
    for _ in range(numChoose):
        choose_route = random.randrange(m)
        n = len(vectors[choose_route])
        if n > 1:
            choose = random.randrange(n - 1)
            for _ in range(numTry):
                try_route = random.randrange(m)
                try_num = random.randrange(len(vectors[try_route]))
                if try_route != choose_route:  # try insert choose before try_num
                    new_vector1 = copy.deepcopy(vectors[choose_route])
                    new_vector1.pop(choose)

                    new_vector2 = copy.deepcopy(vectors[try_route])
                    new_vector2.insert(try_num, vectors[choose_route][choose])
                    if RouteFullness(new_vector1) & RouteFullness(new_vector2):
                        new_dist1 = RouteDistance(new_vector1)
                        new_dist2 = RouteDistance(new_vector2)
                        if new_dist1 + new_dist2 <= vector_lens[choose_route] + vector_lens[try_route]:
                            vectors[choose_route] = new_vector1
                            vectors[try_route] = new_vector2
                            vector_lens[choose_route] = new_dist1
                            vector_lens[try_route] = new_dist2
                            break
                try_route = random.randrange(m)
                try_num = random.randrange(len(vectors[try_route]))
                if try_route != choose_route:  # try to swap
                    new_vector1 = copy.deepcopy(vectors[choose_route])
                    new_vector2 = copy.deepcopy(vectors[try_route])

                    try:
                        new_vector1[choose] = vectors[try_route][try_num]
                        new_vector2[try_num] = vectors[choose_route][choose]
                    except IndexError:
                        print('Oops, IndexError')
                        print(vectors[try_route])
                        print(vectors[choose_route])
                        print(try_num)
                        print(choose)
                    if RouteFullness(new_vector1) & RouteFullness(new_vector2):
                        new_dist1 = RouteDistance(new_vector1)
                        new_dist2 = RouteDistance(new_vector2)
                        if new_dist1 + new_dist2 <= vector_lens[choose_route] + vector_lens[try_route]:
                            vectors[choose_route] = new_vector1
                            vectors[try_route] = new_vector2
                            vector_lens[choose_route] = new_dist1
                            vector_lens[try_route] = new_dist2
                            break

    return vectors, vector_lens


def IntraLocalSearch(vector, numChoose, numTry, vector_len):
    n = len(vector)
    if n > 1:
        for _ in range(numChoose):
            choose = random.randrange(n)
            for _ in range(numTry):
                try_num = random.randrange(n)   # try insert before 'try_num'
                if choose < try_num:
                    el = vector[choose]
                    new_vector = copy.deepcopy(vector)
                    new_vector.insert(try_num, el)
                    new_vector.pop(choose)
                else:
                    el = vector[choose]
                    new_vector = copy.deepcopy(vector)
                    new_vector.pop(choose)
                    new_vector.insert(try_num, el)
                if RouteFullness(new_vector):
                    newDistance = RouteDistance(new_vector)
                    if newDistance <= vector_len:
                        vector = new_vector
                        vector_len = newDistance
            choose = random.randrange(n-1)  # try swap elements
            new_vector = copy.deepcopy(vector)
            new_vector[choose] = vector[choose+1]
            new_vector[choose+1] = vector[choose]

            if RouteFullness(new_vector):
                newDistance = RouteDistance(new_vector)
                if newDistance <= vector_len:
                    vector = new_vector
                    vector_len = newDistance
    return vector, vector_len


def Decode(x2, distance):
    route = {a: [] for a in range(m)}
    x1 = copy.deepcopy(x2)
    n1 = len(x1)
    n = len(x1)
    current_capacity = [0 for _ in range(m)]

    for r in range(m):
        route[r].append(x1[r])
        current_capacity[r] += demands[x1[r]]
    i = m
    while i < n:
        route_distances = [distance[route[r][-1]][x1[i]] for r in range(m)]
        sorted_routes = [(idx, r) for (idx, r) in enumerate(route_distances)]
        # sorted_routes.sort(key=lambda itm: itm[1], reverse=True)
        sorted_routes.sort(key=lambda itm: itm[1])
        assigned = 0
        for j in range(len(sorted_routes)):
            r = sorted_routes[j][0]
            if current_capacity[r] + demands[x1[i]] <= capacity:
                route[r].append(x1[i])
                current_capacity[r] += demands[x1[i]]
                assigned = 1
                break
        if assigned == 0:
            max_route_size = max([len(route[r]) for r in range(m)])
            for k in range(max_route_size-1, 0, -1):
                for r in range(m):
                    route_distances[r] = 0
                    if (k > 0) & (len(route[r]) > k):
                        route_distances[r] += distance[x1[i]][route[r][k-1]]
                    if len(route[r]) - 1 > k:
                        route_distances[r] += distance[route[r][k+1]][x1[i]]
                sorted_routes = [(idx, r) for (idx, r) in enumerate(route_distances)]
                # sorted_routes.sort(key=lambda itm: itm[1], reverse=True)
                sorted_routes.sort(key=lambda itm: itm[1])
                for j in range(0, len(sorted_routes)):
                    r = sorted_routes[j][0]
                    if len(route[r]) > k:
                        if current_capacity[r] + demands[x1[i]] - demands[route[r][k]] <= capacity:
                            current_capacity[r] += demands[x1[r - 1]] - demands[route[r][k]]
                            x1.append(route[r][k])
                            route[r][k] = x1[i]
                            assigned = 1
                            n += 1
                            break
        if assigned == 0:
            route_made = 0
            return route_made, route
        if n > 2*n1:
            return 0, route
        i += 1
    route_made = 1
    return route_made, route


nametask, dimension, coordinates, distance, capacity, demands = distances_cvrp()
start_time = time.time()
time_duration = 60
T = 40
k2 = 0.5
k3 = 0.7

m = int(nametask[nametask.find('k')+1:])
numParticles = dimension // 4

k1 = (0.1 - (1 - sqrt(abs(1 - 2 * time_duration / T)) * 0.5))

globalBestRoute = []
globalBestLength = np.inf

x = {'route': {p: [0 for _ in range(dimension)] for p in range(numParticles)},
     'length': {p: np.inf for p in range(numParticles)},
     'velocity': {p: [] for p in range(numParticles)}}

b = {'route': {p: [] for p in range(numParticles)},
     'length': {p: np.inf for p in range(numParticles)}}

for i in range(numParticles):    # Initial population
    xi = [j for j in range(1, dimension)]
    vi = [j for j in range(1, dimension)]
    shuffle(xi)
    shuffle(vi)
    x['route'][i] = xi
    x['velocity'][i] = [random.randrange(dimension - 1)]
    b[i] = xi

p = {}  # pool of global best solutions
p_len = []

best_in_pool = 0

num_choose = dimension // 5
num_try = dimension // 5

iterations = 0

while (time.time() - start_time <= time_duration) | (iterations <= T):
    localBestRoute = []
    localBestLength = np.inf
    extra_made = 0
    extra_route = {}
    g_encode = []

    while extra_made == 0:
        x_extra = [i for i in range(1, dimension)]
        shuffle(x_extra)
        extra_made, extra_route = Decode(x_extra, distance)
    for particle in range(numParticles):
        routeMade, route = Decode(x['route'][particle], distance)
        if routeMade == 0:
            routeMade = 1
            route = copy.deepcopy(extra_route)
        if routeMade:
            routeAllLength = 0
            route_length = [0 for _ in range(m)]
            for i in range(m):
                route[i], route_length[i] = IntraLocalSearch(route[i], num_choose, num_try, route_length[i])  # Apply Intra local search on all particles
            route, route_length = InterLocalSearch(route, num_choose, num_try, route_length)  # Apply Inter local search on all particles

            for i in range(m):
                route_length[i] = RouteDistance(route[i])
                routeAllLength += route_length[i]

            if routeAllLength < localBestLength:
                localBestRoute = route
                localBestLength = routeAllLength
                g_encode = x['route'][particle]

            if routeAllLength < b['length'][particle]:
                b['route'][particle] = x['route'][particle]
                b['length'][particle] = routeAllLength

    if localBestLength < globalBestLength:
        globalBestRoute = localBestRoute
        globalBestLength = localBestLength
    key = len(p)
    p[key] = localBestRoute
    p_len.append(localBestLength)

    # LS for BestSolutionPool and refining the pool

    p1 = {}  # new pool
    p1_len = []
    temp_route = {i: [] for i in range(m)}
    temp_len = [0 for i in range(m)]

    best_in_pool = p_len.index(min(p_len))

    for particle in range(len(p)):
        for i in range(m):
            particle_route_len = RouteDistance(p[particle][i])
            temp_route[i], temp_len[i] = IntraLocalSearch(p[particle][i], num_choose, num_try,
                                                          particle_route_len)  # Apply Intra local search on global best pool
        temp_len_intra = sum(temp_len)

        p_len_routs = []
        for i in range(m):
            p_len_routs.append(RouteDistance(p[particle][i]))

        temp_route_inter, temp_len_inter = InterLocalSearch(p[particle], num_choose, num_try, p_len_routs)  # Apply Inter local search on global best pool
        temp_len_inter = sum(temp_len_inter)
        if temp_len_intra < temp_len_inter:
            if temp_len_intra < p_len[particle]:
                key = len(p1)
                p1[key] = temp_route
                p1_len.append(temp_len_intra)
            else:
                if particle == best_in_pool:
                    key = len(p1)
                    p1[key] = p[particle]
                    p1_len.append(p_len[particle])
        else:
            if temp_len_inter < p_len[particle]:
                key = len(p1)
                p1[key] = temp_route_inter
                p1_len.append(temp_len_inter)
            else:
                if particle == best_in_pool:
                    key = len(p1)
                    p1[key] = p[particle]
                    p1_len.append(p_len[particle])
        x['route'][particle], x['velocity'][particle] = particleMove(x['route'][particle], x['velocity'][particle], b[particle], g_encode)
    p = p1
    p_len = p1_len
    for particle in range(len(p)):
        if p_len[particle] < globalBestLength:
            globalBestRoute = p[particle]
            globalBestLength = p_len[particle]
    iterations += 1

print(iterations)
print(globalBestLength)
print(p[best_in_pool])
caps = []
for i in range(m):
    caps.append(RouteFullness(p[best_in_pool][i]))
print(caps)

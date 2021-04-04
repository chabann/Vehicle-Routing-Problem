from BasicFunctions import calcValue, answer_loop
import random
import time
from random import shuffle


def stochastic_two_opt(perm):
    randlimit = len(perm) - 1
    c1, c2 = random.randint(0, randlimit), random.randint(0, randlimit)
    exclude = [c1, randlimit if c1 == 0 else c1 - 1, 0 if c1 == randlimit else c1 + 1]

    while c2 in exclude:
        c2 = random.randint(0, randlimit)

    c1, c2 = c2, c1 if c2 < c1 else None
    perm[c1:c2] = perm[c1:c2][::-1]
    return perm


def local_search(best, max_no_improv, neighborhood_size, penalty_koef, dist):
    count = 0
    while count < max_no_improv:
        candidate = {"vector": [v for v in best["vector"]]}

        for _ in range(neighborhood_size):
            stochastic_two_opt(candidate["vector"])

        candidate["cost"], candidate["penalty"] = \
            calcValue(candidate["vector"], dist, penalty_koef)

        if (candidate["cost"] < best["cost"]) & (candidate["penalty"] <= best["penalty"]):
            count, best = 0, candidate
        else:
            count += 1
    return best


def search(neighborhoods, max_no_improv_ls, max_no_improv, time_limit, start_time, penalty_koef, dist):
    best = {}
    dim = len(dist)
    best["vector"] = [i for i in range(0, dim)]
    shuffle(best["vector"])
    best["cost"], best["penalty"] = calcValue(best["vector"], dist, penalty_koef)
    iter_, count = 0, 0

    while (time.time() - start_time) < time_limit:
        for neigh in neighborhoods:

            if (time.time() - start_time) >= time_limit:
                break

            candidate = {"vector": [v for v in best["vector"]]}

            if count > max_no_improv:
                count = 0
                shuffle(candidate["vector"])

            for _ in range(neigh):
                stochastic_two_opt(candidate["vector"])

            candidate["cost"], candidate["penalty"] = \
                calcValue(candidate["vector"], dist, penalty_koef)
            candidate = local_search(candidate, max_no_improv_ls, neigh, penalty_koef, dist)
            iter_ += 1

            if (candidate["cost"] < best["cost"]) & (candidate["penalty"] <= best["penalty"]):
                best, count = candidate, 0
                break
            else:
                count += 1

    return best


def main_vns(coords, dist, time_limit, penalty_koef):
    start_time = time.time()

    numb_neigh = len(coords)

    max_no_improv = 20  # 50
    max_no_improv_ls = 50  # 50
    neighborhoods = list(range(numb_neigh))
    best = search(neighborhoods, max_no_improv_ls, max_no_improv, time_limit, start_time, penalty_koef, dist)

    # print("Done VNS. Best Solution:", best["cost"], "penalty:", best["penalty"])
    # print(time.time() - start_time, 'seconds for VNS')

    best["vector"] = list(answer_loop(best["vector"]))
    return best["vector"], best["cost"]

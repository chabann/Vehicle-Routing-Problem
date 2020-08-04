import numpy as np
import numpy.random as rnd
import random
import matplotlib.pyplot as plt

from distances import distances


def calculateEnergy(x, dis):
    n = len(x)
    energy = 0
    for i in range(n-1):
        energy += dis[x[i]][x[i+1]]
    return energy + dis[x[n-1]][x[0]]


def GenerateStateCandidate(x):
    n = len(x)
    rnd.seed(1000)
    leftbound = random.randint(0, n)
    rightbound = random.randint(0, n)

    if leftbound < rightbound:
        subx0 = x[0: leftbound]
        subx = x[leftbound: rightbound]
        subx1 = x[rightbound: n]
    else:
        subx0 = x[0: rightbound]
        subx = x[rightbound: leftbound]
        subx1 = x[leftbound: n]
    subx = np.flip(subx)
    x1 = np.hstack((subx0, subx, subx1))
    # print("state candidate:", x1)
    return x1


def GetTransitionProbability(E, t):
    return np.exp(-E/t)


def MakeTransit(p):
    value = rnd.sample()
    if value <= p:
        return 1
    else:
        return 0


def DecreaseTemperature(t, i):
    return t * 0.2 / (i + 1)


name, dim, coords, distance = distances()
# print("Coordinates", coords, "dimension: ", dim)

initialTemperature = 1000
endTemperature = 0.000001
iterMax = 50000

state = rnd.permutation(dim)  # задаём вектор начального состояния, как случайную перестановку городов

currentEnergy = calculateEnergy(state, distance)
currentTemp = initialTemperature

for iter in range(iterMax):
    stateCandidate = GenerateStateCandidate(state)   # получаем состояние - кандидат
    candidateEnergy = calculateEnergy(stateCandidate, distance)  # вычисляем его энергию
    if candidateEnergy < currentEnergy:
        currentEnergy = candidateEnergy
        state = stateCandidate
    else:
        probability = GetTransitionProbability(candidateEnergy - currentEnergy, currentTemp)
        if MakeTransit(probability):
            currentEnergy = candidateEnergy
            state = stateCandidate
    currentTemp = DecreaseTemperature(initialTemperature, iter)  # уменьшаем температуру

    if currentTemp <= endTemperature:
        break

state = np.append(state, state[0])
print("state: ", state, "Energy: ", currentEnergy)

coords = np.transpose(coords)

plt.plot(coords[1], coords[2], 'ro')
coords_state = []

for i in range(dim + 1):
    coords_state.append([coords[1][state[i]], coords[2][state[i]]])
    
coords_state = np.transpose(coords_state)
plt.plot(coords_state[0], coords_state[1], 'b-')

plt.grid()
plt.show()



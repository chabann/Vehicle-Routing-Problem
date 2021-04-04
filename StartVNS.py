import matplotlib.pyplot as plt
import numpy as np
import xlwt
from math import inf

from variable_neighborhood_search import main_vns
from BasicFunctions import distances


name, dim, coords, dist = distances()
coords = np.transpose(coords)
# index = name.find(".")
# name = name[0:index+1]
book = xlwt.Workbook()
VNS = book.add_sheet("VNSearch_" + name)

state = []
state_wr = []
value_wr = inf
time_limit = 800
penalty_koeff = np.max(dist)


for iteration in range(1):
    route, cost = main_vns(coords, dist, time_limit, penalty_koeff)

    row_iter = VNS.row(iteration)
    index = 0
    """row_iter.write(index, "Route: ")
    for index, value in enumerate(route):
        row_iter.write(index + 1, int(value))"""
    row_iter.write(index, "Length:  ")
    row_iter.write(index + 1, cost)
    if cost < value_wr:
        value_wr = cost
        state_wr = route
    print(cost)

book.save("Results/VNS_" + name + ".xls")

plt.plot(coords[1], coords[2], 'ro')
coords_state = []

for i in range(dim+1):
    coords_state.append([coords[1][state_wr[i]], coords[2][state_wr[i]]])

coords_state = np.transpose(coords_state)
plt.plot(coords_state[0], coords_state[1], 'b-')

plt.title("VNS  " + name)
plt.grid()
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import xlwt
from math import inf

from SimulatedAnnealing import main


book = xlwt.Workbook()
SA = book.add_sheet("SimulatedAnnealing")

coords = []
state = []
state_wr = []
energy_wr = inf


for iter in range(10):
    state, energy, coords = main()

    row_iter = SA.row(iter)
    index = 0
    row_iter.write(index, "Route: ")
    for index, value in enumerate(state):
        row_iter.write(index + 1, int(value))
    row_iter.write(index + 2, "Length:  ")
    row_iter.write(index + 3, energy)
    if energy < energy_wr:
        energy_wr = energy
        state_wr = state

book.save("Results/SA.xls")

plt.plot(coords[1], coords[2], 'ro')
coords_state = []
dim = len(state_wr)

for i in range(dim):
    coords_state.append([coords[1][state_wr[i]], coords[2][state_wr[i]]])

coords_state = np.transpose(coords_state)
plt.plot(coords_state[0], coords_state[1], 'b-')

plt.grid()
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import xlwt
from math import inf

from SelfOrganizingMap import som_main
from BasicFunctions import distances


name, dim, coords, dist = distances()
coords = np.transpose(coords)
index = name.find(".")
name = name[0:index]
book = xlwt.Workbook()
SOM = book.add_sheet("SelfOrganizingMap_" + name)

route_wr = []
route_length_wr = inf


for iter in range(1):
    route, route_length = som_main()

    row_iter = SOM.row(iter)
    index = 0
    row_iter.write(index, "Route: ")
    for index, value in enumerate(route):
        row_iter.write(index + 1, int(value))
    row_iter.write(index + 2, "Length:  ")
    row_iter.write(index + 3, route_length)
    if route_length < route_length_wr:
        route_length_wr = route_length
        route_wr = route

book.save("Results/SOM_" + name + ".xls")

plt.plot(coords[1], coords[2], 'ro')
coords_route = []

for i in range(dim):
    coords_route.append([coords[1][route_wr[i]], coords[2][route_wr[i]]])

coords_route = np.transpose(coords_route)
plt.plot(coords_route[0], coords_route[1], 'b-')

plt.title("SOM  " + name)
plt.grid()
plt.show()

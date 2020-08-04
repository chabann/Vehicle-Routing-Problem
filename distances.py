from math import sqrt


def distances():
    f = open('task.txt', 'r')
    start_coords = 0
    coordinates = []
    num = 0
    nametask = ""
    dimension = 0
    for line in f:
        line = line.strip()
        if line.find("NAME:") >= 0:
            line1 = line.split(' ')
            nametask = line1[-1]
        elif line.find("DIMENSION:") >= 0:
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

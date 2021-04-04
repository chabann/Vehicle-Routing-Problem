# from config import SOMConfig as Config
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from BasicFunctions import distances, answer_loop, sum_way
# import util

window_size = 5
dpi = 100
node_radius = 0.1
# b_init = 10
b_growth = 1.005

b_init = 10.0
alpha = 0.03
b_ceil = 1000
# mu = 1.0
mu = 0.6
iter_lim = 800
record_moment = np.arange(0, iter_lim, 10)
record = True


def dist(p1, p2):
    return np.linalg.norm(p1 - p2)


def g_func(djj_star: int, l: float, g: float) -> float:
    if djj_star < l:
        return np.exp(-djj_star ** 2 / g ** 2)
    return 0.0


def calc_champ_node(band_array: np.array, city: np.array) -> int:
    dist_array = np.array([dist(node, city) for node in band_array])
    # find min value index
    return dist_array.argmin()


def update_node(node: np.array, city: np.array, djj_star: int, node_num,  beta: float) -> np.array:
    delta_node = mu * g_func(djj_star, node_num * 0.2, beta) * (city - node)
    return delta_node


def update_band(
    band_array: np.array, city: np.array, j_star: int, node_num, beta: float
) -> np.array:
    new_band_array = band_array.copy()
    for j in range(node_num):
        djj_star = np.amin([np.abs(j - j_star), node_num - np.abs(j - j_star)])
        new_band_array[j, :] += update_node(band_array[j, :], city, djj_star, node_num, beta)
    return new_band_array


def make_way_from_circle(band_array: np.array, city: np.array, np_distances):
    way = []
    for i in range(len(band_array)):
        lenght_evkl = np.inf
        li = -1
        xi = band_array[i, 0]
        yi = band_array[i, 1]
        for j in range(len(city)):
            xj = city[j, 0]
            yj = city[j, 1]
            if ((xi - xj)**2 + (yi - yj)**2)**(1/2) < lenght_evkl:
                lenght_evkl = ((xi - xj)**2 + (yi - yj)**2)**(1/2)
                li = j
        way.append(li)
    new_way = []
    for i in way:
        if i not in new_way:
            new_way.append(i)
    distance = sum_way(new_way, np_distances)
    new_way = answer_loop(new_way)
    return new_way, distance


def som_begin(band_array: np.array, city_array: np.array, np_distances, city_num, node_num):
    beta = b_init
    np.random.shuffle(city_array)
    way_best = np.inf
    sol_best = []
    if record:
        for i in range(iter_lim):
            # if i in record_moment:
            # filename = "iteration-" + str(i) + ".png"
            # file_path = dir_name + filename
            # plt.savefig(file_path)

            picked_city = city_array[i % city_num, :]
            j_star = calc_champ_node(band_array, picked_city)
            band_array = update_band(band_array, picked_city, j_star, node_num, beta)
            circle_band = np.vstack((band_array, band_array[0, :]))
            # if i % 10 == 0:
                # plt.title("iteration=" + str(i + 1))
                # elastic_band.set_data(circle_band[:, 0], circle_band[:, 1])
                # plt.pause(0.001)
            beta = np.amin([b_ceil, beta * b_growth])
            # beta = (1 - alpha) * beta

        # plt.show()
        way, summary_distance = make_way_from_circle(band_array, city_array, np_distances)
        print(way, summary_distance)
        return way, summary_distance

    """else:
        i = 1
        while plt.get_fignums():
            picked_city = city_array[i % city_num, :]
            j_star = calc_champ_node(band_array, picked_city)
            band_array = update_band(band_array, picked_city, j_star, beta)
            circle_band = np.vstack((band_array, band_array[0, :]))
            plt.title("iteration=" + str(i))
            elastic_band.set_data(circle_band[:, 0], circle_band[:, 1])
            i += 1
            # beta = np.amin([b_ceil, beta * b_growth])
            beta = (1 - alpha) * beta
            plt.pause(0.001)
    """


def som_main():
    nametask, city_num, cities, np_distances = distances()
    np_cities = np.zeros((city_num, 2))
    np_dist = np.array((np_distances))
    for i in range(city_num):
        np_cities[i, :] = cities[i][1:3]
    width_x = np.max(np_cities[:, 0]) - np.min(np_cities[:, 0])
    width_y = np.max(np_cities[:, 1]) - np.min(np_cities[:, 1])
    width = np.amax([width_x, width_y])
    """np_cities[:, 0] -= np.min(np_cities[:, 0])
    np_cities[:, 0] /= width
    np_cities[:, 1] -= np.min(np_cities[:, 1])
    np_cities[:, 1] /= width"""
    center_x = np.average(np_cities[:, 0])
    center_y = np.average(np_cities[:, 1])
    figsize = (window_size, window_size)

    node_num = int(city_num * 4)
    angles = np.linspace(0, 2 * np.pi, node_num)
    np_band = np.array(
        [
            width_x/10 * np.sin(angles) + center_x,
            width_y/10 * np.cos(angles) + center_y,
        ]
    ).transpose()
    # fig = plt.figure(figsize=figsize, dpi=dpi)
    rgb = [[0, 0, 0], [0.3, 0, 0], [0.45, 0, 0], [0.6, 0, 0], [0.75, 0, 0], [0.85, 0, 0],
           [0, 0.3, 0], [0, 0.45, 0], [0, 0.6, 0], [0, 0.75, 0], [0, 0.85, 0],
           [0, 0, 0.3], [0, 0, 0.45], [0, 0, 0.6], [0, 0, 0.75], [0, 0, 0.85]]
    # plt.scatter(np_cities[:, 0], np_cities[:, 1], s=50, marker="*")
    # elastic_band, = plt.plot(np_band[:, 0], np_band[:, 1])
    # plt.title("iteration=" + str(0))
    # plt.grid()
    # plt.pause(0.001)
    return som_begin(np_band, np_cities, np_distances, city_num, node_num)

from random import randint
import numpy as np
from itertools import permutations, combinations
import matplotlib.pyplot as plt
from timeit import default_timer as timer




def calc_distance(p1,p2):
    distance = np.sqrt(abs(((cities_array[p1][1]-cities_array[p2][1])**2 + (cities_array[p1][2]-cities_array[p2][2]**2))))
    return round(distance,3)

def get_inputs():
    num_of_cities = int(input("What is the number of cities: "))
    x_size = int(input("What is the x length of the map: "))
    y_size = int(input("What is the y length of the map: "))
    return (num_of_cities, x_size, y_size)

def generate_random_coord(cities_data):
    cities_array = np.empty((0, 3), int)
    for i in range(cities_data[0]):
        x_coord = randint(0, cities_data[1])
        y_coord = randint(0, cities_data[2])
        cities_array = np.vstack((cities_array,[i, x_coord, y_coord]))

    return cities_array

def generate_permutation(cities_array):
    for p in permutations(cities_array[:, 0]):

        if p[0] < p[-1]:  # if p's first digit is lower than that of its reverse, keep it, if not, do not keep it

            possible_paths.append(p)
    return possible_paths

def calc_best_path(possible_paths):
    travel_costs = list()
    for path in possible_paths:

        total_path = 0
        for i in range(1, len(path)):
            distance_calculated = calc_distance(path[i - 1], path[i])
            total_path += distance_calculated

        travel_costs.append([total_path, path])
    return travel_costs

def plot_cities(cities_data, cities_array):
    plt.plot([cities_array[:,1]], [cities_array[:,2]],'ro')
    plt.axis([0, cities_data[1]+5, 0, cities_data[2]+5])
    print (cities_array)
    plt.show()


cities_data = get_inputs() # gets num of cities and map coord
start = timer() # start counting time
cities_array = generate_random_coord(cities_data) # generated data for cities
#possible_paths = ([x for x in permutations(cities_array[:,0])])
possible_paths = list()
# plotting
plot_cities(cities_data, cities_array)

possible_paths = generate_permutation(cities_array) # generates permutations
travel_costs = calc_best_path(possible_paths)

best_travel_costs = min(travel_costs)
print(travel_costs)

stop = timer()
script_time = stop-start;
print("Best travel path was: {}. Time takes was: {}".format(best_travel_costs,script_time))





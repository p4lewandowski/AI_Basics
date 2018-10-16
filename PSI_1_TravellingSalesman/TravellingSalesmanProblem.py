from random import randint
import numpy as np
from itertools import permutations, combinations, chain
import matplotlib.pyplot as plt
from timeit import default_timer as timer


def get_inputs():
    # num_of_cities = int(input("What is the number of cities: "))
    # x_size = int(input("What is the x length of the map: "))
    # y_size = int(input("What is the y length of the map: "))
    # return (num_of_cities, x_size, y_size)

    return (11, 100, 100)

def generate_random_coord(cities_data):
    cities_array = np.empty((0, 3), int)
    for i in range(cities_data[0]):
        x_coord = randint(0, cities_data[1])
        y_coord = randint(0, cities_data[2])
        cities_array = np.vstack((cities_array,[i, x_coord, y_coord]))

    return cities_array

def generate_permutation(cities_array):

    possible_paths = list()
    for p in permutations(cities_array[:, 0]):
        # limiting useless permutations
        if p[1] < p[-1]:  # if p's first digit is lower than that of its reverse, keep it, if not, do not keep it
            if p[0] == 0:
                p = p + (0,) # end at 0 location
                possible_paths.append(p)

    return possible_paths

def calc_distance(p1,p2, cities_array):
    distance = np.sqrt(abs((cities_array[p1][1]-cities_array[p2][1])**2 + (cities_array[p1][2]-cities_array[p2][2])**2))
    return round(distance,3)

def calc_best_path(possible_paths, cities_array):
    travel_costs = list()
    for path in possible_paths:

        total_path = 0
        for i in range(1, (len(path))):
            distance_calculated = calc_distance(path[i - 1], path[i], cities_array)
            total_path += distance_calculated
            #print("i-th iteration: {}   total_path = {}   ".format(i, total_path))
        travel_costs.append([total_path, path])
    return travel_costs

def plot_cities(cities_data, cities_array, best_travel_costs):
    proper_path_xcoord = list(); proper_path_ycoord = list()
    plt.plot([cities_array[:,1]], [cities_array[:,2]],'ro')
    plt.axis([0, cities_data[1]+5, 0, cities_data[2]+5])
    # take all combination from x,y coordinates
    # chain - take coord of them and separate
    # zip - make them tuples
    plt.plot(
        *zip(*chain.from_iterable(combinations(cities_array[:,[1, 2]],2))),
        color='brown', marker='o')

    for i in best_travel_costs:
            proper_path_xcoord.append(cities_array[i,1])
            proper_path_ycoord.append(cities_array[i,2])
    plt.plot(proper_path_xcoord, proper_path_ycoord, '-', linewidth =3)
    plt.plot(proper_path_xcoord[0], proper_path_ycoord[0], marker = 'o', markerfacecolor = 'yellow', markersize = 10)
    plt.show()


def brute_force_travellingsalesman(cities_data, cities_array):

    start = timer() # start counting time
    #possible_paths = ([x for x in permutations(cities_array[:,0])])
    possible_paths = generate_permutation(cities_array) # generates permutations
    travel_costs = calc_best_path(possible_paths, cities_array)
    best_travel_costs = min(travel_costs)
    stop = timer()
    script_time = stop-start;

    print("Best travel path was: {0} with distance {1:0.3f}. Time takes was: {2}".format(best_travel_costs[1], best_travel_costs[0],script_time))
    plot_cities(cities_data, cities_array, best_travel_costs[1])

def nearest_neigbour_travellingsalesman(cities_data, cities_array):
    order = [0]
    complete_distance = 0
    start_nn = timer() # start counting time

    for i, c in enumerate(cities_array):
        distance, order = nearest_neigbour_closest_city(cities_array, order)
        complete_distance +=distance

    stop_nn = timer()
    script_time_nn = stop_nn-start_nn;
    print("Best travel path for nearest-neighbour was: {0} with distance {1:.3f}. Time takes was: {2}".format(order, complete_distance,script_time_nn))
    plot_cities(cities_data, cities_array, order)


def nearest_neigbour_closest_city(cities_array, visited):
    best_proposition = 0
    best_current_distance = 0
    for i, c in enumerate(cities_array):
        if i not in visited:
            distance_calculated = np.sqrt(abs((cities_array[i][1]-cities_array[visited][-1][1])**2 +
                                              (cities_array[i][2]-cities_array[visited][-1][2])**2))

            if distance_calculated < best_current_distance:
               best_current_distance = distance_calculated
               best_proposition = i
            elif best_current_distance == 0:
               best_current_distance = distance_calculated
               best_proposition = i
       # if last element equal to lenght of cities_array calculate distance to 1st city
        elif best_current_distance == 0 and i == (len(cities_array)-1):
             best_current_distance = np.sqrt(abs((cities_array[visited][i][1] - cities_array[0][1]) ** 2 +
                                              (cities_array[visited][i][2] - cities_array[0][2]) ** 2))



    visited.append(best_proposition)
    return best_current_distance, visited

def initialize_problem():
    # Generate the data
    cities_data = get_inputs() # gets num of cities and map coord
    # cities array = city_id, xcoord, yccord
    cities_array = generate_random_coord(cities_data) # generated data for cities

    # Brute force
    brute_force_travellingsalesman(cities_data, cities_array)

    # Nearest neighbour
    nearest_neigbour_travellingsalesman(cities_data, cities_array)

if __name__ == "__main__":
    initialize_problem()



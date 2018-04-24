import numpy as np
import random

population_size = 100
DNA_size = 8
cycles = 5
population = ["" for x in range(population_size)]

for i in range(0, len(population)):
    population[i] = ''.join(["%s" % random.randint(0, 1) for num in range(0, DNA_size)])
    #i = int(i,2)
    #adaptation_grade = 2*(i*i+1)

print(population)

crossing_probability = 0.3
mutation_probability = 0.7

possible_pairs = list()
possible_indexes = np.arange(0, population_size, 1)

x_copy = possible_indexes[:]  # copy
random.shuffle(x_copy)
shuffled_pairs = [x_copy[i * 2: (i + 1) * 2] for i in range(population_size // 2)]

sequence = 0
while sequence < cycles:
    sequence +=1
    for i in range (0, len(population)//2):
        if (random.uniform(0, 1) < crossing_probability):
            locus = random.randint(1, DNA_size-2) # not whole series
            temp_copy = population[shuffled_pairs[i][0]] #copy needed
            population[shuffled_pairs[i][0]] = population[shuffled_pairs[i][0]][:locus] + population[shuffled_pairs[i][1]][locus:]
            population[shuffled_pairs[i][1]] = temp_copy[:locus] + temp_copy[locus:]

    for i in range(0, len(population)):
        if (random.uniform(0, 1) < mutation_probability):
            locus = random.randint(0, DNA_size-1)
            population[i] = population[i][:locus] + str(1 - int(population[i][locus])) + population[i][locus+1:]
    for i in range(0, population_size):
        

    print (population)



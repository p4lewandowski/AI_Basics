import numpy as np
import random
import matplotlib.pyplot as plt

population_size = 10
DNA_size = 7 #sie of binary string
cycles = 1000
population = ["" for x in range(population_size)] #declaring empty array of size population
crossover_probability = 0.75
mutation_probability = 0.05
accomplishment_value = 0 # condition fo terminate cycles
#arrays for plot
data = np.array(list([]))
sequence_col = np.array(list([]))
data_col = np.array(list([]))

for i in range(0, population_size): population[i] = ''.join(["%s" % random.randint(0, 1) for num in range(0, DNA_size)]) #create string of 0s and 1s
possible_pairs = list()
possible_indexes = np.arange(0, population_size, 1)
print(population)
###### Begin the process ######
sequence = 0
while sequence < cycles:
    new_population = ["" for x in range(population_size)]
    total_val = 0
    sequence +=1
    probability_offset = 0
    accomplishment_value = 0

    ###### Calculate total value for roulette ######
    population_unique, population_counts = np.unique(population, return_counts=True)
    occurences_dictionary = dict(zip(population_unique, population_counts))
    for value in occurences_dictionary:
        temp_val = 2 * ((int(value, 2) ** 2 + 1))
        total_val +=  temp_val # calculate total value

    ###### Zakresy Ruletki ######
    occurences_dictionary_adaptation = dict()
    for value in list(occurences_dictionary):
        adaptation_grade = probability_offset + (2 * ((int(value, 2)) ** 2 + 1))/total_val  # prob_off + (2*(x**2_1)/sum
        probability_offset = adaptation_grade #increase the range upwards
        occurences_dictionary_adaptation[value] = adaptation_grade

    ###### Ruletka ######
    for index in range(0, population_size):
        locus = random.uniform(0, 1)
        for new_population_member in list(occurences_dictionary_adaptation): # list of possible options
            chosen_one = new_population_member;
            if locus < occurences_dictionary_adaptation[new_population_member]: # jesli wylosowana liczba nizsza niz procentowy udzial jakiejs sekwencji binarnej
                new_population[index] = chosen_one
                break # break if smaller

    population = new_population

    ###### Krzyzowanie ######
    random.shuffle(possible_indexes)
    shuffled_pairs = [possible_indexes[i * 2: (i + 1) * 2] for i in range(population_size // 2)]
    for i in range (0, population_size//2):
        if (random.uniform(0, 1) < crossover_probability):
            locus = random.randint(1, DNA_size-1)
            temp_copy = population[shuffled_pairs[i][0]] #copy needed
            population[shuffled_pairs[i][0]] = population[shuffled_pairs[i][0]][:locus] + population[shuffled_pairs[i][1]][locus:]
            population[shuffled_pairs[i][1]] = temp_copy[:locus] + temp_copy[locus:]

    ###### Mutacja ######
    for i in range(0, population_size): # from i to N
        if (random.uniform(0, 1) < mutation_probability): # if mutation
            locus = random.randint(0, DNA_size-1) #span of range
            population[i] = population[i][:locus] + str(1 - int(population[i][locus])) + population[i][locus+1:]

    ###### check how far to accomplush #####
    ###### win condition ######
    for i in range(0, population_size):
        accomplishment_value += (int(population[i], 2))
    accomplishment_condition = accomplishment_value / population_size

    if sequence % 1 == 0:
        sequence_col = np.append(sequence_col, sequence)
        data_col = np.append(data_col, accomplishment_condition)
    #### if acomplished - break %%%%
    if accomplishment_condition >=125:
         break


data = np.vstack([sequence_col, data_col])

print("Cycle number: " + str(sequence) + " " + str(new_population))
#np.savetxt('Sequence{}Population{}.txt'.format(sequence,population_size), data, newline='\n')
plt.plot([data[0]], [data[1]], '-ro')
plt.show()


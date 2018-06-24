"""
@autor Wildo Monges
This program resolve the problems Santa's Bag posted by Kaggle https://www.kaggle.com/c/santas-uncertain-bags
I implemented using genetic algorithm
Note:
    Run it typing python santa.py
Example of the result:
    Printing population
    horse: 5.086419
    ball: 1.831313
    bike: 36.083249
    train: 10.868049
    coal: 23.440352
    book: 0.647851
    doll: 7.829742
    blocks: 9.742349
    gloves: 3.128789
"""

import random
import numpy as np
import pandas as pd
import csv

""" All constants used in this program"""
TOYS_NAME_LIST = ['horse', 'ball', 'bike', 'train', 'coal', 'book', 'doll', 'blocks', 'gloves']
MAX_TOY_BY_BAG = 9
MAX_BAG = 1000
MAX_GENERATIONS = 500
MIN_TOYS_BY_BAG = 3
MAX_POUND = 50
MUTATION_PROB = 50
NEW_POPULATION_PROB = 10
MIN_BAGS_SIZE = 1

class Toy:
    """This class is to manipulate Toy like an object"""
    def __init__(self, name, number):
        self.name = name
        self.number = number
        self.description = self.name + '_' + str(self.number)
        self.weight = 0


class RandomWeight:
    """This class has a function to return random weight values"""
    def value(self, name):
        if name == 'horse':
            return self.__horse()
        elif name == 'ball':
            return self.__ball()
        elif name == 'bike':
            return self.__bike()
        elif name == 'train':
            return self.__train()
        elif name == 'coal':
            return self.__coal()
        elif name == 'book':
            return self.__book()
        elif name == 'doll':
            return self.__doll()
        elif name == 'blocks':
            return self.__blocks()
        elif name == 'gloves':
            return self.__gloves()

    """Private functions"""
    def __horse(self):
        return max(0, np.random.normal(5, 2, 1)[0])

    def __ball(self):
        return max(0, 1 + np.random.normal(1, 0.3, 1)[0])

    def __bike(self):
        return max(0, np.random.normal(20, 10, 1)[0])

    def __train(self):
        return max(0, np.random.normal(10, 5, 1)[0])

    def __coal(self):
        return 47 * np.random.beta(0.5, 0.5, 1)[0]

    def __book(self):
        return np.random.chisquare(2, 1)[0]

    def __doll(self):
        return np.random.gamma(5, 1, 1)[0]

    def __blocks(self):
        return np.random.triangular(5, 10, 20, 1)[0]

    def __gloves(self):
        return 3.0 + np.random.rand(1)[0] if np.random.rand(1) < 0.3 else np.random.rand(1)[0]


class Bag:
    """This class allow to work over bags"""
    def __init__(self):
        self.id = None
        self.toys = []
        self.weight = 0
        self.toy_counter = 0

    @staticmethod
    def bags_into_rule_toys(bags):
        """Check if the bags is into the rules"""
        for bag in bags:
            if bag.toy_counter < MIN_TOYS_BY_BAG or bag.toy_counter > MAX_TOY_BY_BAG or bag.weight > MAX_POUND:
                return False
        return True


class Evolution:
    """This class implements methods to apply genetic algorithms"""
    @staticmethod
    def generate_population(number_population):
        """Method to generate a population of weights grouped by Toy"""
        rw = RandomWeight()
        weights_by_toy = {}
        for t in TOYS_NAME_LIST:
            weights = {}
            for i in range(number_population):
                weight = rw.value(t)
                weights[i] = weight
            weights_by_toy[t] = weights  # {'home':{0: w0,1:w1}, }
        return weights_by_toy

    @staticmethod
    def generate_bags(populations, lenght, toys_data):
        """Method to generate the bags that are into the rules"""
        bags_population = {}
        toys_counter = {}
        bag_index = 0
        for i in range(lenght):
            bags = []
            toy_index = 0
            while len(bags) < MAX_BAG:
                bag = Bag()
                bag.id = bag_index
                bag.weight = 0
                bag.toys = []
                aux_toys = []
                bag.toy_counter = 0
                while (bag.toy_counter < MAX_TOY_BY_BAG) and (toy_index < amount_toys) and (
                        bag.weight + populations[toys_data[toy_index].name][i]) <= MAX_POUND:
                    toy = toys_data[toy_index]
                    toy.weight = populations[toy.name][i]
                    aux_toys.append(toy)
                    bag.toy_counter = bag.toy_counter + 1
                    bag.weight = bag.weight + toy.weight
                    toy_index = toy_index + 1
                bag_index = bag_index + 1
                bag.toys = aux_toys
                bags.append(bag)
            bags_population[i] = bags
            toys_counter[i] = toy_index
        return bags_population, toys_counter

    @staticmethod
    def mutate(populations, arr_popu_index):
        """Function to mutate random weight of the weight of populations"""
        rw = RandomWeight()
        mutation_amount = random.randint(1, len(arr_popu_index))
        for i in range(mutation_amount):
            index_to_mutate = random.randint(0, len(arr_popu_index))
            for toy in TOYS_NAME_LIST:
                populations[toy][index_to_mutate] = rw.value(toy)

        return populations

    @staticmethod
    def print_population(populations, index):
        """Print population in format"""
        print('Printing population')
        for t in TOYS_NAME_LIST:
            print("%s: %f" % (t, populations[t][index]))


"""Start resolving the problem"""

# Load toys from file
toys = pd.read_csv('gifts.csv', index_col=None)
amount_toys = toys['GiftId'].count()
index = 0
toys_data = []
while index < amount_toys:
    arr = toys['GiftId'][index].split()
    name_number = arr[0].split('_')
    name = name_number[0]
    number = name_number[1]
    t = Toy(name, number)
    toys_data.append(t)
    index = index + 1

# Generate 50 Random Population, you can change it if the result is not found into this number
number_population = 50
generation_counter = 0
found_result = False
populations = {}
selected_bags = []

# My first population of weights grouped by toy
populations = Evolution.generate_population(number_population)
# Generate bags based of the population
bags_by_population, toys_counter = Evolution.generate_bags(populations, number_population, toys_data)

selected_population = []
while not found_result and generation_counter < MAX_GENERATIONS:
    print('Generation # %i' % (generation_counter + 1))
    selected_population_index = 0
    for pindex, bags in bags_by_population.items():
        # select the better populations
        if len(bags) == MAX_BAG and toys_counter[pindex] == amount_toys and Bag.bags_into_rule_toys(bags):
            selected_bags = bags
            found_result = True
            break
        elif len(bags) == MAX_BAG and toys_counter[pindex] == amount_toys and not Bag.bags_into_rule_toys(bags):
            selected_population.append(pindex)
            selected_population_index = selected_population_index + 1

    # Finish the while if the result is found
    if found_result:
        break

    # mutation
    if random.randint(0, 100) >= MUTATION_PROB and len(selected_population) >= MIN_BAGS_SIZE:
        populations = Evolution.mutate(populations, selected_population)
    # generate new population
    elif random.randint(0, 100) >= NEW_POPULATION_PROB:
        populations = Evolution.generate_population(number_population)

    # Generate new bags based on the new populations values
    bags_by_population, toys_counter = Evolution.generate_bags(populations, number_population, toys_data)
    generation_counter = generation_counter + 1
    selected_population = []

# Write result.txt
header = True
for bag in selected_bags:
    names = []
    for toy in bag.toys:
        names.append(toy.description)

    with open("result.csv", "a") as f:
        if header:
            f.write('Gifts\n')
            header = False
        f.write(' '.join(names) + '\n')

print("Printed Weights found in generation #%i" % (generation_counter))
print('Done')
Evolution.print_population(populations, selected_population_index)

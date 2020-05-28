from organism import *
import numpy as np
import math
from PIL import Image, ImageOps
import random
import csv

class Algorithm():
    def __init__(self, goal, w, h, num_poly, num_vertex, comparison_method, savepoints, outdirectory):
        self.goal = goal
        self.goalpx = np.array(goal)
        self.w = w
        self.h = h
        self.num_poly = num_poly
        self.num_vertex = num_vertex
        self.comparison_method = comparison_method
        self.data = []
        self.savepoints = savepoints
        self.outdirectory = outdirectory

    def save_data(self, row):
        # function to be called every generation to remember data on that generation
        self.data.append(row)

    def write_data(self):
        # writes all collected data to a file
        with open(self.outdirectory + '/data.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            for row in self.data:
                writer.writerow(row)

class SA(Algorithm):
    # https://am207.github.io/2017/wiki/lab4.html
    def __init__(self, goal, w, h, num_poly, num_vertex, comparison_method, savepoints, outdirectory, iterations):
        super().__init__(goal, w, h, num_poly, num_vertex, comparison_method, savepoints, outdirectory)
        self.iterations = iterations

        # initializing organism
        self.best = Organism(0,0,None, self.w, self.h)
        self.best.initialize_genome(self.num_poly, num_vertex)
        self.best.genome_to_array()
        self.best.calculate_fitness_mse(self.goalpx)

        self.current = deepcopy(self.best)

        # define data header for SA
        self.data.append(["Polygons", "Generation", "bestMSE", "currentMSE"])


    def acceptance_probability(self, dE, T):
        return np.exp(-dE/T)

    def cooling_sigmoid(self, i):
        t = 1000/(1 + np.exp(0.00001 * (i - 500000)))
        return t

    def run(self):
        for i in range(1, self.iterations):
            james = Organism(0, i, None, self.w, self.h)
            james.genome = self.current.deepish_copy_genome()
            james.random_mutation(1)
            james.genome_to_array()

            james.calculate_fitness_mse(self.goalpx)

            dE = james.fitness - self.current.fitness
            T = self.cooling_sigmoid(i)

            acceptance = self.acceptance_probability(dE, T)

            if random.random() < acceptance:
                # self.current = deepcopy(james)
                self.current.genome = james.deepish_copy_genome()
                self.current.fitness = james.fitness
                self.current.genome_to_array()

            if self.current.fitness < self.best.fitness:
                self.best.genome = james.deepish_copy_genome()
                self.best.fitness = james.fitness
                self.best.genome_to_array()
                # self.best = deepcopy(self.current)

            if i in self.savepoints:
                self.best.generation = i
                self.best.save_img(self.outdirectory)
                self.best.save_polygons(self.outdirectory)
                self.save_data([self.num_poly, i, self.best.fitness, self.current.fitness])

        self.best.generation = i
        self.best.save_img(self.outdirectory)
        self.best.save_polygons(self.outdirectory)
        self.save_data([self.num_poly, i, self.best.fitness, self.current.fitness])

import numpy as np
import pygmo as pg
import csv

class sphere_function:
    def __init__(self, dim):
        self.dim = dim
        self.objectives = np.zeros((200, 2))
        self.counter = 0
        count = 0
        with open('pareto_solutions_v01.csv', 'rt') as f:
            reader = csv.reader(f)
            for row in reader:
                obj1 = float(row[0])
                obj2 = float(row[1])
                self.objectives[count][0] = obj1 / 600
                self.objectives[count][1] = obj2
                #accuracy = np.append(accuracy, acc)

                #print(obj1, obj2)

                count += 1

            f.close()

    def fitness(self, x):
        #print(self.counter)
        #f1 = sum(x**2)
        #f2 = sum(x*x*x + 40*x)
        #print(type(f1))
        #f1 = np.float64(np.random.rand(1))
        #f2 = np.float64(np.random.rand(1))
        f1 = self.objectives[self.counter][0]
        f2 = self.objectives[self.counter][1]
        #print(self.objectives[self.counter][0], self.objectives[self.counter][1])
        self.counter += 1
        return (f1, f2, )

    def get_bounds(self):
        return ([-1] * self.dim, [1] * self.dim)

    def get_nobj(self):
        return 2

    def get_name(self):
        return "Custom Function"

    def get_extra_info(self):
        return "\tDimensions: " + str(self.dim)

prob = pg.problem(sphere_function(3))

#print(prob)

algo = pg.algorithm(pg.moead(gen = 1))
#algo = pg.algorithm(pg.bee_colony(gen=20, limit=20))
pop = pg.population(prob=prob, size=100)
pop = algo.evolve(pop)
#print(pop.get_f())

from matplotlib import pyplot as plt
ax = pg.plot_non_dominated_fronts(pop.get_f())

#plt.plot(ax)

#plt.ylabel('Traveled path $l_1$')
#plt.xlabel('Lateral velocity $l_2$')
#plt.title("Objective space with a Pareto front")

#plt.savefig('pareto_solutions.png')

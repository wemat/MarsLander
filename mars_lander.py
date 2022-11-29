#%matplotlib inline  #uncomment in jupyter
#from IPython import display  #uncomment in jupyter
import numpy as np
from random import choice
from random import randrange
from random import randint
from random import uniform
from matplotlib import pyplot as plt
from itertools import zip_longest
import random
import time
import pandas as pd


# from functools import lru_cache
# @lru_cache(maxsize=None)

def calc_positions(thrust_vector, x, y, vv, vh, dv, dh):
    """calculates the next x,y-coordinates of the rocket given the current position,speed, thrust and angle"""
    x_new, y_new = x, y
    g = -3.711
    res_x = []
    res_y = []
    res_vv = []
    res_vh = []
    for i in range(0, len(thrust_vector[0])):
        pwr = thrust_vector[0][i]
        angle = thrust_vector[1][i]

        a_v = np.sin(np.radians(90 - angle)) * (pwr)  # acceleration vertical (derrived of power vector)
        a_h = np.cos(np.radians(90 - angle)) * (-pwr)  # acceleration horizontal (derrived of power vector)
        dh = (vh) + (0.5 * a_h)
        dv = (vv) + (0.5 * (g + a_v))

        vv += g + a_v
        vh += a_h

        x_new, y_new = x_new + dh, y_new + dv
        res_x.append(int(x_new))
        res_y.append(int(y_new))
        res_vv.append(vv)
        res_vh.append(vh)

    return [res_x, res_y, res_vv, res_vh, *thrust_vector]


def generate_thrust_vector(n, pwr=0, angle=0):
    """generates a random landing vector"""
    res_pwr = []
    res_angles = []
    for i in range(n):
        pwr = choice([min(pwr + 1, 4), max(0, pwr - 1), pwr])
        res_pwr.append(pwr)

        angle = randrange(angle - 15, angle + 15)
        res_angles.append(max(min(angle, 90), -90))
    return [res_pwr, res_angles]


def old_valid_vec(surface, chromosome):
    """checks if shuttle crashes into surface"""
    idx = (np.abs(np.asarray(surface[0]) - chromosome[0][-1])).argmin()
    if surface[1][idx] > chromosome[1][-1] or (idx == 0 and surface[0][idx] > chromosome[0][-1]):
        return False
    return True


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


# Return true if line segments AB and CD intersect
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def valid_vec(surface, A, B):
    if B[0] < surface[0][0] or B[0] > surface[0][-1]:
        return False

    px, py = 0, 0
    for x, y in zip(*surface):
        if intersect(A, B, (px, py), (x, y)):
            return False
        px, py = x, y
    return True


def create_population(pop_size, chrom_size, x, y, vv, vh, surface, first_generation_plus, dv=0, dh=0):
    pop = []
    i = 0
    while i < (pop_size + first_generation_plus):
        chromosome = calc_positions(generate_thrust_vector(chrom_size + randint(-5, 5)), x, y, vv, vh, dv, dh)
        if valid_vec(surface, (chromosome[0][0], chromosome[1][0]), (chromosome[0][-1], chromosome[1][-1])):
            pop.append(chromosome)
        i += 1
    print(len(pop))
    return pop


class MarsLander:
    def __init__(self, pop_size, chrom_size, x, y, vv, vh, surface, first_generation_plus, hbreak_multiplier,
                 vbreak_multiplier,
                 non_linear_score, graded_retain_perc, non_graded_retain_perc, crossover_prob, mutation_prob,
                 time_limit):
        self.pop_size = pop_size
        self.chrom_size = chrom_size
        self.x = x
        self.y = y
        self.vv = vv
        self.vh = vh
        self.surface = surface
        self.flat_surface = self.get_flat_surf()
        self.total_dist = round(
            np.mean([np.linalg.norm(np.array(xy) - np.array((self.x, self.y))) for xy in self.flat_surface]))
        self.last_population = []

        # hyperparameters
        self.first_generation_plus = first_generation_plus
        self.hbreak_multiplier = hbreak_multiplier
        self.vbreak_multiplier = vbreak_multiplier
        self.non_linear_score = non_linear_score
        self.graded_retain_perc = graded_retain_perc
        self.non_graded_retain_perc = non_graded_retain_perc
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.time_limit = time_limit

    def update(self, x, y, vv, vh):
        self.x, self.y, self.vv, self.vh = x, y, vv, vh

    def get_flat_surf(self):
        return [xy for i in range(1, len(self.surface[1])) for xy in
                [[self.surface[0][i - 1], self.surface[1][i - 1]], [self.surface[0][i], self.surface[1][i]]] if
                self.surface[1][i] == self.surface[1][i - 1]]

    def set_best_dist(self, x, y):
        self.best_dist = round(np.mean([np.linalg.norm(np.array(xy) - np.array((x, y))) for xy in self.flat_surface]))

    def score(self, x, y, vv, vh, pwr, angle):
        """punish: euclidean distance to x meteres above landing zone"""
        dist = round(
            np.median([np.linalg.norm(np.array([x, y]) - np.array([xy[0], xy[1] + 50])) for xy in self.flat_surface]))
        inv_dist = (1 + ((self.total_dist - dist) / self.total_dist)) ** self.non_linear_score  # the closer the higher
        # print(inv_dist)

        return (
                dist +
                self.hbreak_multiplier * inv_dist * np.abs(vh) +
                self.vbreak_multiplier * inv_dist * np.abs(vv) +
                max(0, self.flat_surface[0][1] - y)  # for high ground landing
        )

    def accept(self, x, y, vv, vh, pwr, angle):
        flat_x = [xy[0] for xy in self.flat_surface]
        # print(flat_surface[0][1]+300)
        if min(flat_x) <= x <= max(flat_x) and self.flat_surface[0][1] + 10 < y and self.flat_surface[0][1] + 50 > y:
            if np.abs(vv) <= 40 and np.abs(vh) <= 20:
                return True
        return False

    def get_gene(self, chrom, idx):
        return (chrom[0][idx], chrom[1][idx], chrom[2][idx], chrom[3][idx], chrom[4][idx], chrom[5][idx])

    def selection(self, chromosomes_list):
        GRADED_RETAIN_PERCENT = self.graded_retain_perc  # percentage of retained best fitting individuals
        NONGRADED_RETAIN_PERCENT = self.non_graded_retain_perc  # percentage of retained remaining individuals (randomly selected)
        sorted_chroms = sorted(chromosomes_list, key=lambda x: self.score(*self.get_gene(x, -1)))
        idx = int(len(sorted_chroms) * GRADED_RETAIN_PERCENT)
        random_cnt = int(len(sorted_chroms) * NONGRADED_RETAIN_PERCENT)
        # print([random.choice(sorted_chroms[idx:]) for i in range(random_cnt)])
        return sorted_chroms[:idx] + [random.choice(sorted_chroms[idx:]) for i in range(random_cnt)]

    def crossover(self, parent1, parent2):
        return [self.weighted_vector(np.array(parent1[0]), np.array(parent2[0]), pwr=True),
                self.weighted_vector(np.array(parent1[1]), np.array(parent2[1]))]

    def __mutation(self, child):
        """mutates a chromosome: either change an existing tuple(thrust/angle) or appending a new one (extending the landing vector)"""
        if uniform(0, 1) > 0.7:
            index = random.choice([i for i in range(len(child[0]))])
            diff_pwr = np.insert(child[0][1:] - child[0][:-1], 0, child[0][0])
            diff_pwr[index] = choice([-1, 0, 1])
            diff_angle = np.insert(child[1][1:] - child[1][:-1], 0, child[1][0])
            diff_angle[index:] += choice([-5, 0, 5])
            # print(child)
            return [cumsum(np.array(diff_pwr)), cumsum(np.array(diff_angle), lb=-90, hb=90)]

        prev_pwr = child[0][-1]
        pwr = choice([min(prev_pwr + 1, 4), max(0, prev_pwr - 1), prev_pwr])
        prev_angle = child[1][-1]
        angle = randrange(prev_angle - 15, prev_angle + 15)
        return [np.append(child[0], pwr), np.append(child[1], max(min(angle, 90), -90))]

    def mutation(self, child):
        """mutates a chromosome: either change an existing tuple(thrust/angle) or appending a new one (extending the landing vector)"""
        if uniform(0, 1) > 0.5:
            index = random.choice([i for i in range(len(child[0]))])
            diff_pwr = np.insert(child[0][1:] - child[0][:-1], 0, child[0][0])
            diff_pwr[index] = choice([-1, 0, 1])
            diff_angle = np.insert(child[1][1:] - child[1][:-1], 0, child[1][0])
            diff_angle[index:] += choice([-5, 0, 5])
            # print(child)
            return [cumsum(np.array(diff_pwr)), cumsum(np.array(diff_angle), lb=-90, hb=90)]

        for i in range(randint(1, 3)):
            prev_pwr = child[0][-1]
            pwr = choice([min(prev_pwr + 1, 4), max(0, prev_pwr - 1), prev_pwr])
            prev_angle = child[1][-1]
            angle = randrange(prev_angle - 15, prev_angle + 15)
            child[0] = np.append(child[0], pwr)
            child[1] = np.append(child[1], max(min(angle, 90), -90))

        return [child[0], child[1]]

    def new_mutation(self, child, pwr_dict={0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3]}):
        """mutates a chromosome: either change an existing tuple(thrust/angle) or appending a new one (extending the landing vector)"""
        if uniform(0, 1) > 0.7:
            for i in range(1):
                index = randint(1, len(child[0]) - 1)
                if index == len(child[0]) - 1:
                    return [np.append(child[0][:index], choice(pwr_dict[child[0][index]])),
                            np.append(child[1][:index], randrange(child[1][index - 1] - 15, child[1][index - 1] + 15))]

                child[0][index] = choice(pwr_dict[child[0][index]])
                child[1][index:] = randrange(child[1][index - 1] - 15, child[1][index - 1] + 15)
                child[0] = np.append(child[0][:index], smoother(child[0][index:], 1))
                child[1] = np.append(child[1][:index], smoother(child[1][index:], 15))
            return [child[0], child[1]]

        for i in range(randint(1, 3)):
            prev_pwr = child[0][-1]
            pwr = choice([min(prev_pwr + 1, 4), max(0, prev_pwr - 1), prev_pwr])
            prev_angle = child[1][-1]
            angle = randrange(prev_angle - 15, prev_angle + 15)
            child[0] = np.append(child[0], pwr)
            child[1] = np.append(child[1], max(min(angle, 90), -90))

        return [child[0], child[1]]

    def generation(self, population):
        select = self.selection(population)
        children = []

        while len(children) < (self.pop_size - len(select)):
            ## crossover
            parent1 = random.choice(select)[4:]
            parent2 = random.choice(select)[4:]
            child = self.crossover(parent1, parent2)

            if uniform(0, 1) > (1 - self.mutation_prob):
                ## mutation
                child = self.mutation(child)

            # chrom = [[x..],[y..],[vv...],[vh...],[pwr..],[angls...]]
            child = calc_positions(child, self.x, self.y, self.vv, self.vh, dv=0, dh=0)
            if valid_vec(self.surface, (child[0][0], child[1][0]), (child[0][-1], child[1][-1])):
                select.append(child)
            else:
                select.append([c[:-2] for c in child])

                # print("XXXX")
        # return new generation
        # print(len(children))
        return select + children

    def genetic_algorithm(self, x, y, vv, vh, debug=False):
        if not self.last_population:
            population = create_population(self.pop_size, self.chrom_size, x, y, vv, vh, self.surface,
                                           self.first_generation_plus)
        else:
            population = self.last_population
        answer = []
        start = time.time()
        end = 0

        while not answer and (end - start) < self.time_limit:
            # print("pop ",i)
            population = self.generation(population)
            if debug:
                best = population[0]
                plt.title("vertical Velocity :{} horizontal Velocity : {} Score: {}".format(
                    round(best[2][-1]),
                    round(best[3][-1]),
                    round(self.score(*self.get_gene(best, -1)))))
                plt.plot(best[0], best[1])
                plt.plot(self.surface[0], self.surface[1], 'black')

                display.display(plt.gcf())
                display.clear_output(wait=True)

            for chrom in population:
                if self.accept(*self.get_gene(chrom, -1)):
                    answer.append(chrom)
            end = time.time()

        if not answer:
            self.last_population = [[gene[1:] for gene in chrom] for chrom in population]
            return population[0], False
        else:
            return answer, True

    def weighted_vector(self, parent1_vec, parent2_vec, pwr=False):
        diff1 = np.insert(parent1_vec[1:] - parent1_vec[:-1], 0, parent1_vec[0])
        diff2 = np.insert(parent2_vec[1:] - parent2_vec[:-1], 0, parent2_vec[0])
        weight = uniform(0 + self.crossover_prob, 1 - self.crossover_prob)
        if pwr:
            return cumsum(np.array([int((p1g * weight) + p2g * (1 - weight)) for p1g, p2g in zip(diff1, diff2)]))
        return np.cumsum([int((p1g * weight) + p2g * (1 - weight)) for p1g, p2g in zip(diff1, diff2)])


def cumsum(array, lb=0,hb=4):
    result = np.zeros(array.size)
    result[0] = array[0]
    for k in range(1, array.size):
        result[k] = max(lb, min(hb,result[k-1]+array[k]))
    return result

def smoother(list,max_diff):
    i=0
    if i+2>len(list):
        return list

    while np.abs(list[i]-list[i+1])>max_diff:
        if list[i+1]>list[i]:
            list[i+1]-=1
        else:
            list[i+1]+=1
        if np.abs(list[i]-list[i+1])==max_diff:
            i+=1
        if i+2>len(list):
            return list
    return list

# the tests
inital_values= {0:(2500,2700,0,0),
                1:(6500,2800,0,-100),
                2:(6500,2800,0,-90),
                3:(500,2700,0,100),
                4:(6500,2700,0,-50)}

surfaces = {0:[[0, 1000, 1500, 3000, 4000, 5500, 6999], [100, 500, 1500, 1000, 150, 150, 800]],
            1:[[0, 1000, 1500, 3000, 3500, 3700, 5000, 5800, 6000, 6999], [100, 500, 100, 100, 500, 200, 1500, 300, 1000, 2000]],
            2: [[0, 1000, 1500, 3000, 4000, 5500, 6999], [100, 500, 1500, 1000, 150, 150, 800]],
            3: [[0, 300, 350, 500, 800, 1000, 1200, 1500, 2000, 2200, 2500, 2900, 3000, 3200, 3500, 3800, 4000, 5000, 5500, 6999], [1000, 1500, 1400, 2000, 1800, 2500, 2100, 2400, 1000, 500, 100, 800, 500, 1000, 2000, 800, 200, 200, 1500, 2800]],
            4:[[0, 300, 350, 500, 1500, 2000, 2500, 2900, 3000, 3200, 3500, 3800, 4000, 4200, 4800, 5000, 5500, 6000, 6500, 6999], [1000, 1500, 1400, 2100, 2100, 200, 500, 300, 200, 1000, 500, 800, 200, 800, 600, 1200, 900, 500, 300, 500]]
            }


pop_size = 20
chrom_size = 30
test_i = 1

x, y, vv, vh = inital_values[test_i]  # vv = velocity vertical / vh = velocity horizontal
surface = surfaces[test_i]
first_generation_plus = 30
hbreak_multiplier = 2  # 20
vbreak_multiplier = 4  # 10
non_linear_score = 3
graded_retain_perc = 0.2
non_graded_retain_perc = 0
crossover_prob = 0.3
mutation_prob = 0.7

TIME_LIMIT = 0.1

mars_lander = MarsLander(pop_size, chrom_size, x, y, vv, vh, surface, first_generation_plus, hbreak_multiplier,
                         vbreak_multiplier, non_linear_score, graded_retain_perc, non_graded_retain_perc,
                         crossover_prob, mutation_prob, time_limit=TIME_LIMIT)
start = time.time()
for i in range(80):
    # print(i,x,y)
    res, is_valid = mars_lander.genetic_algorithm(x, y, vv, vh, debug=True)

    plt.clf()
    if is_valid:
        break

    x, y, vv, vh = res[0][0], res[1][0], res[2][0], res[3][0]
    mars_lander.update(x, y, vv, vh)

end = time.time()

plt.clf()
display.display(plt.gcf())
display.clear_output(wait=True)

plt.title(str(len(res)) + " Solutions after " + str(round(end - start, 1)) + " Seconds")
for chrom in res:
    plt.plot(chrom[0], chrom[1])
plt.plot(surface[0], surface[1], 'black')
plt.show()

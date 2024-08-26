import pygame
import sys
import time
import math
import copy
import ast
import numpy as np
from nn_model import NeuralNetwork
from nn_model import mutate_weights


car_size = (20, 40)  # width, length
green = (0, 100, 0)
red = (100, 0, 0)
BorderColor = (255, 255, 255)

win = pygame.display.set_mode((1920, 1080))
# win = pygame.display.set_mode((1120, 1080))
bg = pygame.image.load('track3.png')

class Car:
    def __init__(self):
        self.surface = pygame.Surface((car_size[0], car_size[1]), pygame.SRCALPHA)  # shape of car
        self.surface.fill(red)

        self.position = [960, 950]
        self.angle = 90
        self.speed = 0


        self.corners = []  # position of corners
        self.alive = True

        self.sensors = []  # sensor lines, [(x, y), distance]

        self.distance = 0
        self.time = 0
        self.sum_speed = [[], 0]

    def rotate(self, direction):  # True left, False right
        angle = 1.8
        if not direction:
            angle = -angle
        self.angle += angle
        self.angle %= 360

    def accelerate(self, direction):  # True forward, False backward
        speed = 0.1
        if direction:
            if self.speed < 6:
                self.speed += speed
        else:
            if self.speed > 0:
                self.speed -= speed
        self.move()

    def move(self):
        dx = math.sin(math.radians(self.angle)) * self.speed
        dy = math.cos(math.radians(self.angle)) * self.speed

        self.position[0] += dx
        self.position[1] += dy

    def slow_down(self):
        if self.speed > 0:
            self.speed -= 0.05
            if self.speed < 0:
                self.speed = 0
            self.move()
        elif self.speed < 0:
            self.speed += 0.05
            if self.speed > 0:
                self.speed = 0
            self.move()

    def update(self):
        keys = pygame.key.get_pressed()

        # rotating
        if keys[pygame.K_LEFT]:
            self.rotate(True)
        if keys[pygame.K_RIGHT]:
            self.rotate(False)

        # moving
        if keys[pygame.K_UP]:
            self.accelerate(True)
        elif keys[pygame.K_DOWN]:
            self.accelerate(False)
        else:
            self.slow_down()

        # update time and distance

        if self.alive:
            self.distance += self.speed
            self.time += 1
            self.sum_speed[1] += 1
            self.sum_speed[0].append(self.speed)
        # print(self.time, self.distance)

        # corners check, assume its square because I hate trigonometry
        length = 0.5 * car_size[0]

        left_top = [self.position[0] + math.cos(math.radians(360 - (self.angle + 30))) * length, self.position[1] + math.sin(math.radians(360 - (self.angle + 30))) * length]
        right_top = [self.position[0] + math.cos(math.radians(360 - (self.angle + 150))) * length, self.position[1] + math.sin(math.radians(360 - (self.angle + 150))) * length]
        left_bottom = [self.position[0] + math.cos(math.radians(360 - (self.angle + 210))) * length, self.position[1] + math.sin(math.radians(360 - (self.angle + 210))) * length]
        right_bottom = [self.position[0] + math.cos(math.radians(360 - (self.angle + 330))) * length, self.position[1] + math.sin(math.radians(360 - (self.angle + 330))) * length]

        self.corners = [left_top, left_bottom, right_top, right_bottom]
        self.check_corners()

        # sensor check
        self.sensors = []
        for angle in range(-90, 120, 45):
            self.update_sensor(angle)

    def draw(self):
        rotated_surface = pygame.transform.rotate(self.surface, self.angle)
        rotate_center = rotated_surface.get_rect(center=self.position)

        self.draw_sensor()
        win.blit(rotated_surface, rotate_center)

    def update_sensor(self, angle):
        length = 0
        x = int(self.position[0] + math.sin(math.radians(self.angle + angle))*length)
        y = int(self.position[1] + math.cos(math.radians(self.angle + angle))*length)

        while not bg.get_at((x, y)) == BorderColor and length < 450:
            length += 1
            x = int(self.position[0] + math.sin(math.radians(self.angle + angle))*length)
            y = int(self.position[1] + math.cos(math.radians(self.angle + angle))*length)

        distance = math.sqrt(math.pow(x-self.position[0], 2) + math.pow(y-self.position[1], 2))
        self.sensors.append([(x, y), distance])

    def draw_sensor(self):
        for sensor in self.sensors:
            end_position = sensor[0]
            pygame.draw.line(win, green, self.position, end_position, 2)
            pygame.draw.circle(win, green, end_position, 5)

    def check_corners(self):
        for corner in self.corners:
            if bg.get_at((int(corner[0]), int(corner[1]))) == BorderColor:
                self.alive = False
                break

    def is_alive(self):
        return self.alive

    def get_data(self):
        data = [sensor[1] for sensor in self.sensors]
        return data

    def get_time(self):
        return self.time

    def get_speed(self):
        return sum(self.sum_speed[0]) / len(self.sum_speed[0])
        # return self.speed

    def get_distance(self):
        return (sum(self.sum_speed[0]) * 0.4 / len(self.sum_speed[0])) * self.time
        # return self.distance


# initialize global variables
current_generation = 1
population_size = 30
neural_networks = [NeuralNetwork(5, 6, 4) for _ in range(population_size)]
best_neural_network = NeuralNetwork(5, 6, 4)
distances = [2000, 0]
mutation_strength = 0.01


# def tournament_selection(fitness_scores, k):
#     selected_neural_networks_index = np.random.randint(0, len(fitness_scores), size=k)
#     selected_neural_networks = []
#     for i in selected_neural_networks_index:
#         selected_neural_networks.append(fitness_scores[i])
#     selected_neural_networks.sort(key=lambda x: x[0], reverse=True)
#     return selected_neural_networks[0][1]
#
#
# def select_neural_networks(cars, networks):
#     fitness_scores = [(cars[i].get_distance(), networks[i]) for i in range(population_size)]
#     return [tournament_selection(fitness_scores, 2) for _ in range(population_size)]
#
#
# def create_child_networks(selected_nodes, mutation_strength):
#     for i in range(population_size):
#         weights = selected_nodes[i].get_weights()
#         mutation = mutate_weights(weights, 0.01, mutation_strength)
#         selected_nodes[i].set_weights(mutation)
#     return selected_nodes

def select_neural_networks(cars, networks):
    fitness_scores = [(cars[i].get_distance(), networks[i]) for i in range(population_size)]
    fitness_scores.sort(key=lambda x: x[0], reverse=True)
    # print(fitness_scores[0][0], fitness_scores[1][0], fitness_scores[2][0])
    return [i[1] for i in fitness_scores]


def create_child_networks(chosen_nn, mutation_strength):
    new_neural_networks = []

    num_chosen = 30
    while len(new_neural_networks) < population_size:
        for i in range(num_chosen):
            for _ in range(3):
                if len(new_neural_networks) < population_size:
                    cloned_nn = copy.deepcopy(chosen_nn[i])
                    new_neural_networks.append(cloned_nn)

    for i in range(population_size):
        weights = new_neural_networks[i].get_weights()
        mutation = mutate_weights(weights, 0.05, mutation_strength)
        new_neural_networks[i].set_weights(mutation)
    return new_neural_networks


def run_simulation():
    pygame.init()

    clock = pygame.time.Clock()

    # Track simulation time
    start_time = time.time()
    max_simulation_time = 40

    global current_generation
    global neural_networks
    global best_neural_network
    global distances
    global mutation_strength
    font = pygame.font.SysFont('Arial', 24)

    # initialize cars and neural networks
    cars = [Car() for _ in range(population_size)]

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        win.blit(bg, (0, 0))

        # check if cars still alive and draw them
        still_alive = 0
        for i, car in enumerate(cars):
            if car.is_alive():
                still_alive += 1
                car.update()
                car.draw()

        img1 = font.render("current generation: " + str(current_generation), True, (100, 0, 0))
        img2 = font.render("still alive: " + str(still_alive), True, (100, 0, 0))
        win.blit(img1, (20, 20))
        win.blit(img2, (20, 60))

        current_time = time.time() - start_time

        speed = [0]
        for car in cars:
            if car.is_alive():
                speed.append(car.speed)
        # print(max(speed))
        if still_alive == 0 or current_time > max_simulation_time or (current_time > 3 and max(speed) < 1):
            break


        # cars are taking actions
        for i, car in enumerate(cars):
            data = car.get_data()
            neural_networks[i].forward_pass(data)
            output = neural_networks[i].get_binary_output()
            if output[0] == 1:
                car.accelerate(True)
            if output[1] == 1:
                car.accelerate(False)
            if output[2] == 1:
                car.rotate(True)
            if output[3] == 1:
                car.rotate(False)
        pygame.display.flip()
        clock.tick(60)

    maxa = 0
    b = 0
    for c, car in enumerate(cars):
        if car.get_distance() > b:
            b = car.get_distance()
            maxa = c
    # print(neural_networks[maxa].debug_forward_pass())
    # print(cars[maxa].get_time(),cars[maxa].get_speed(), cars[maxa].get_distance())
    print(mutation_strength)
    current_distances = []
    for car in cars:
        current_distances.append(car.get_distance())

    if max(current_distances) - max(distances) < 2:
        if mutation_strength < 20:
            mutation_strength *= 1.4
        distances = current_distances.copy()
    elif mutation_strength > 0.05:
        distances = current_distances.copy()
        mutation_strength = 0.01
    # print(mutation_strength)
    # print(neural_networks[0].get_weights())
    chosen_nn = select_neural_networks(cars, neural_networks)
    # print(f'car: {cars[2].get_distance()}, nn: {neural_networks[2].get_weights()}')
    print('max car: ' + str(max([car.get_distance() for car in cars])))
    # for i in range(len(cars)):
    #     if cars[i].get_distance() == max([car.get_distance() for car in cars]):
    #         print(f'max nn: {neural_networks[i].get_weights()}')
    neural_networks = create_child_networks(chosen_nn, mutation_strength)
    best_neural_network = neural_networks[0]
    current_generation += 1


def run_track(weights):
    pygame.init()
    clock = pygame.time.Clock()
    neural_network = NeuralNetwork(5, 4, 4)
    print(weights)
    print('----------------')
    neural_network.set_weights(weights)
    print(neural_network.get_weights())
    car = Car()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        win.blit(bg, (0, 0))
        if car.is_alive():
            car.update()
            car.draw()

        data = car.get_data()
        neural_network.forward_pass(data)
        output = neural_network.get_binary_output()
        if output[0] == 1:
            car.accelerate(True)
        if output[1] == 1:
            car.accelerate(False)
        if output[2] == 1:
            car.rotate(True)
        if output[3] == 1:
            car.rotate(False)

        pygame.display.flip()
        clock.tick(60)


for i in range(500):
    run_simulation()
    if i % 25 == 0:
        f = open('best_weight.txt', 'w')
        f.write(str(best_neural_network.get_weights()))
        f.close()
print(best_neural_network.get_weights())

f = open('best_weight.txt', 'r')
x = f.read()
while True:
    run_track(eval(x, {"array": np.array}))



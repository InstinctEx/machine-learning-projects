import random
import numpy as np

class Package:
    def __init__(self, weight, length, width):
        self.weight = weight
        self.length = length
        self.width = width

class Truck:
    def __init__(self, max_weight, length, width):
        self.max_weight = max_weight
        self.length = length
        self.width = width
        self.current_weight = 0
        self.packages = []

    def can_add_package(self, package):
        return (self.current_weight + package.weight <= self.max_weight and
                package.length <= self.length and
                package.width <= self.width)

    def add_package(self, package):
        if self.can_add_package(package):
            self.packages.append(package)
            self.current_weight += package.weight
            return True
        return False

class Particle:
    def __init__(self, packages, trucks):
        self.packages = packages
        self.trucks = trucks
        self.position = [-1] * len(packages)
        self.best_position = list(self.position)
        self.velocity = [random.uniform(-1, 1) for _ in self.packages]
        self.best_fitness = -1

    def update_velocity(self, global_best_position):
        w = 0.5
        c1 = 1.0
        c2 = 2.0
        for i in range(len(self.velocity)):
            r1 = random.random()
            r2 = random.random()
            cognitive_velocity = c1 * r1 * (self.best_position[i] - self.position[i])
            social_velocity = c2 * r2 * (global_best_position[i] - self.position[i])
            self.velocity[i] = w * self.velocity[i] + cognitive_velocity + social_velocity

    def update_position(self):
        for i in range(len(self.position)):
            if random.random() < (1 / (1 + np.exp(-self.velocity[i]))):
                self.position[i] = (self.position[i] + 1) % len(self.trucks)

    def calculate_fitness(self):
        for truck in self.trucks:
            truck.packages.clear()
            truck.current_weight = 0
        for i, package in enumerate(self.packages):
            truck_index = self.position[i]
            if truck_index >= 0:
                self.trucks[truck_index].add_package(package)
        fitness = sum(len(truck.packages) for truck in self.trucks)
        return fitness

def get_user_input():
    num_packages = int(input("Enter the number of packages: "))
    packages = []
    for _ in range(num_packages):
        weight = float(input("Enter the weight of the package: "))
        length = float(input("Enter the length of the package: "))
        width = float(input("Enter the width of the package: "))
        packages.append(Package(weight, length, width))

    num_trucks = int(input("Enter the number of trucks: "))
    trucks = []
    for _ in range(num_trucks):
        max_weight = float(input("Enter the maximum weight of the truck: "))
        truck_length = float(input("Enter the length of the truck: "))
        truck_width = float(input("Enter the width of the truck: "))
        trucks.append(Truck(max_weight, truck_length, truck_width))

    return packages, trucks

def pso(packages, trucks, num_particles=30, max_iter=100):
    particles = [Particle(packages, trucks) for _ in range(num_particles)]
    global_best_position = [0] * len(packages)
    global_best_fitness = -1

    for _ in range(max_iter):
        for particle in particles:
            fitness = particle.calculate_fitness()
            if fitness > particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = list(particle.position)
            if fitness > global_best_fitness:
                global_best_fitness = fitness
                global_best_position = list(particle.position)

        for particle in particles:
            particle.update_velocity(global_best_position)
            particle.update_position()

    return global_best_position, global_best_fitness

def main():
    packages, trucks = get_user_input()
    best_position, best_fitness = pso(packages, trucks)
    print("Best Position:", best_position)
    print("Best Fitness:", best_fitness)

if __name__ == "__main__":
    main()

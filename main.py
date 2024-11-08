import math
import sys
import neat
import pygame
from config import Width, Height, CarSizeX, CarSizeY, BorderColor

Currentgeneration = 0


class Car:
    def __init__(self):
        self.sprite = pygame.image.load('car.png').convert()  # Convert speeds up display rendering
        self.sprite = pygame.transform.scale(self.sprite, (CarSizeX, CarSizeY))
        self.rotated_sprite = self.sprite

        self.position = [409, 118] #Brazil Track
        # self.position = [1285, 164] #Las Vegas Track
        self.angle = 0
        self.speed = 0
        self.speed_set = False  

        
        self.center = [self.position[0] + CarSizeX / 2, self.position[1] + CarSizeY / 2]  
        self.alive = True  
        self.corners = []  

        self.radars = []  
        self.drawing_radars = []  
        self.distance = 0  
        self.time = 0  
        self.stuck_counter = 0  

    def draw(self, screen):
        screen.blit(self.rotated_sprite, self.position)
        self.draw_radar(screen)

    def draw_radar(self, screen):
        for radar in self.radars:
            position = radar[0]
            pygame.draw.line(screen, (0, 255, 0), self.center, position, 1)
            pygame.draw.circle(screen, (0, 255, 0), position, 5)

    def check_collision(self, game_map):
        self.alive = True
        self.corners = [
            (self.position[0], self.position[1]),  
            (self.position[0] + CarSizeX, self.position[1]),  
            (self.position[0], self.position[1] + CarSizeY),  
            (self.position[0] + CarSizeX, self.position[1] + CarSizeY)  
        ]
        for point in self.corners:
            if game_map.get_at((int(point[0]), int(point[1]))) == BorderColor:
                self.alive = False
                break

    def check_radar(self, degree, game_map):
        length = 0
        x = int(self.center[0] + math.sin(math.radians(270 + (self.angle + degree))) * length)
        y = int(self.center[1] + math.cos(math.radians(270 + (self.angle + degree))) * length)

        while not game_map.get_at((x, y)) == BorderColor and length < 300:
            length += 1
            x = int(self.center[0] + math.sin(math.radians(270 + (self.angle + degree))) * length)
            y = int(self.center[1] + math.cos(math.radians(270 + (self.angle + degree))) * length)

        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars.append([(x, y), dist])

    def update(self, game_map):
        if not self.speed_set:
            self.speed = 10
            self.speed_set = True

        self.rotated_sprite = self.rotate_center(self.sprite, self.angle)

        self.position[0] += math.cos(math.radians(180 - self.angle)) * self.speed
        self.position[1] += math.sin(math.radians(180 - self.angle)) * self.speed

        self.position[0] = max(self.position[0], 20)
        self.position[0] = min(self.position[0], Width - CarSizeX - 20)
        self.position[1] = max(self.position[1], 20)
        self.position[1] = min(self.position[1], Height - CarSizeY - 20)

        self.distance += self.speed
        self.time += 1
        self.center = [int(self.position[0]) + CarSizeX / 2, int(self.position[1]) + CarSizeY / 2]
        self.corners = self.calculate_corners(0.5 * CarSizeX)
        self.check_collision(game_map)
        self.radars.clear()
        for d in range(-90, 120, 45):
            self.check_radar(d, game_map)

    def calculate_corners(self, length):
        left_top = [self.center[0] + math.cos(math.radians(180 - (self.angle + 30))) * length,
                    self.center[1] + math.sin(math.radians(180 - (self.angle + 30))) * length]
        right_top = [self.center[0] + math.cos(math.radians(180 - (self.angle + 150))) * length,
                     self.center[1] + math.sin(math.radians(180 - (self.angle + 150))) * length]
        left_bottom = [self.center[0] + math.cos(math.radians(180 - (self.angle + 210))) * length,
                       self.center[1] + math.sin(math.radians(180 - (self.angle + 210))) * length]
        right_bottom = [self.center[0] + math.cos(math.radians(180 - (self.angle + 330))) * length,
                        self.center[1] + math.sin(math.radians(180 - (self.angle + 330))) * length]
        return [left_top, right_top, left_bottom, right_bottom]

    def get_data(self):
        return [int(radar[1] / 30) for radar in self.radars] + [0] * (5 - len(self.radars))

    def is_alive(self):
        return self.alive

    def get_reward(self):
        if self.speed < 2:
            return -5  
        return self.distance / (CarSizeX / 2) + self.speed_factor()

    def speed_factor(self):
        if self.speed > 15:
            return 0.1  
        elif self.speed < 10:
            return -0.1  
        return 0  

    def rotate_center(self, image, angle):
        rect = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rect.center = rotated_image.get_rect().center
        return rotated_image.subsurface(rect).copy()


def run_simulation(genomes, config):
    nets, cars = [], []

    pygame.init()
    screen = pygame.display.set_mode((Width, Height))

    for _, g in genomes:
        net = neat.nn.RecurrentNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0
        cars.append(Car())

    clock = pygame.time.Clock()
    game_map = pygame.image.load('BrazilTrack.png').convert()
    # game_map = pygame.image.load('LasVegasTrack.png').convert()
    game_map = pygame.transform.scale(game_map, (Width, Height))

    counter = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        for i, car in enumerate(cars):
            output = nets[i].activate(car.get_data())
            choice = output.index(max(output))
            if choice == 0:
                car.angle += 15
            elif choice == 1:
                car.angle -= 15
            elif choice == 2:
                car.speed = max(car.speed - 2, 12)
            else:
                car.speed += 2

        still_alive = 0
        for i, car in enumerate(cars):
            if car.is_alive():
                still_alive += 1
                car.update(game_map)
                genomes[i][1].fitness += car.get_reward()

        if still_alive == 0 or counter >= 30 * 40:
            break

        screen.blit(game_map, (0, 0))
        for car in cars:
            if car.is_alive():
                car.draw(screen)

        pygame.display.flip()
        clock.tick(60)
        counter += 1


if __name__ == "__main__":
    try:
        config_path = "./config.txt"
        config = neat.config.Config(neat.DefaultGenome,
                                    neat.DefaultReproduction,
                                    neat.DefaultSpeciesSet,
                                    neat.DefaultStagnation,
                                    config_path)
    except FileNotFoundError:
        print(f"Error: The configuration file at '{config_path}' was not found.")
        sys.exit(1)

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.run(run_simulation, 1000)
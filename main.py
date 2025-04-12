import numpy as np
import pygame
from pygame.locals import *

from Neural_Network import NeuralNetwork, NeuralLayer, ActivationLayer
from Neural_Functions import NeuralFunctions

import random
import copy

pygame.init()

clock = pygame.time.Clock()

num_of_birds = 100

generation = 1

pipe_freq = 2000
last_pipe = -pipe_freq

prediction_frames_max = 5

screen_width = 900
screen_height = 600

bw = 30
bh = 30

ground_pos = 0
ground_speed = 3

score = 0

game_over = False

screen = pygame.display.set_mode((screen_width, screen_height))

run = True

background = pygame.image.load('images\\background image.jpg')
background = pygame.transform.smoothscale(background, screen.get_size())

ground = pygame.image.load('images\\ground.png')
ground = pygame.transform.scale(ground, (1000, 60))

neuron_max_difference = 1 / 100

pipes_to_pass = []

bird_start_x = screen_width / 8


def display_text(text, font_size, color, x, y):
    font = pygame.font.Font(None, font_size)
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    text_rect.topleft = (x, y)
    screen.blit(text_surface, text_rect)


class Data:

    def __init__(self, pos, next_pipe_b_l, next_pipe_b_r, next_pipe_t):
        _, height = pos
        pipe_l, _ = next_pipe_b_l
        pipe_r, _ = next_pipe_b_r
        _, pipe_b = next_pipe_b_r
        _, pipe_t = next_pipe_t

        self.top_diff = height - bh / 2 - pipe_t
        self.bottom_diff = pipe_b - (height + bh / 2)
        self.dist_l = pipe_l - bird_start_x + bw / 2
        self.dist_r = pipe_r - bird_start_x + bw / 2


class Brain:

    def __init__(self):

        self.model = NeuralNetwork(loss=NeuralFunctions.cross_entropy_loss,
                                   loss_diff=NeuralFunctions.cross_entropy_loss_diff,
                                   learning_rate=0.05, iters=100)

        self.model.add_layer(NeuralLayer(4, 2))
        self.model.add_layer(ActivationLayer(NeuralFunctions.relu_activation, NeuralFunctions.relu_activation_diff))
        self.model.add_layer(NeuralLayer(2, 2))
        self.model.add_layer(ActivationLayer(NeuralFunctions.softmax, lambda x: 1))

    def normalize_data(self, data):
        ret = [data[0] / screen_height, data[1] / screen_height, data[2] / screen_width, data[3] / screen_width]
        return ret

    def train(self, data):
        input_data = []
        output = []

        for state in data:
            input_data.append(self.normalize_data([state.top_diff, state.bottom_diff, state.dist_l, state.dist_r]))
            cond = state.bottom_diff <= 50

            output.append([[0, 1] if cond else[1, 0]])

        input_data = np.array(input_data)
        output = np.array(output).squeeze(axis=1)

        self.model.fit(input_data, output)

    def predict(self, state):
        input_data = np.array([self.normalize_data([state.top_diff, state.bottom_diff, state.dist_l, state.dist_r])])
        prediction = self.model.predict(input_data)[0].argmax()
        return prediction

    def randomize(self):
        for layer in self.model.layers:
            if not hasattr(layer, 'weights'):
                continue

            random_arr = np.random.uniform(-neuron_max_difference, neuron_max_difference, size=len(layer.weights))
            layer.weights = layer.weights + random_arr[:, np.newaxis]
            

class Bird(pygame.sprite.Sprite):
    def __init__(self, x, y, use_AI=True):
        super().__init__()

        self.index = 0
        self.cnt = 0
        self.num_sprites = 3
        self.last_click = 10000
        self.max_ground_time = 5
        self.ground_time = 0

        self.use_ai = use_AI

        self.gameover = False
        self.prediction_frame = 0

        self.life_time = 0

        self.images = [pygame.transform.scale(pygame.image.load(f'images\\bird{idx}.png'), (bw, bh))
                       for idx in range(self.num_sprites)]

        self.image = self.images[0]
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)

        self.vel = 0.0

        self.all_states = []
        self.brain = Brain()

    def update(self):

        least_vel = -10.0
        flap_cooldown = 5
        gravity = 0.65
        speed = 6

        time_between_clicks = 5

        cur_state = None

        if pygame.sprite.spritecollide(self, pipes, False):
            self.gameover = True

        if self.rect.bottom >= screen_height - ground.get_height() + 20 and self.vel < 0:
            self.ground_time += 1
            if self.ground_time >= self.max_ground_time:
                self.gameover = True
        else:
            self.ground_time = 0

        if not game_over:
            self.life_time += 1

        if not self.gameover:
            cur_state = Data(self.rect.center, pipes_to_pass[0].rect.topleft,
                             pipes_to_pass[0].rect.topright, pipes_to_pass[1].rect.bottomleft)
            self.all_states.append(cur_state)

        self.last_click += 1

        self.prediction_frame = (self.prediction_frame + 1) % prediction_frames_max
        prediction = 0

        if not self.gameover and self.prediction_frame == 0 and self.use_ai:
            prediction = self.brain.predict(cur_state)

        if self.gameover:
            self.rect.x -= ground_speed

        if not self.gameover and (prediction or pygame.key.get_pressed()[K_SPACE]) and self.last_click > time_between_clicks:

            self.vel = speed
            self.last_click = 0

        self.rect.y -= self.vel
        self.rect.y = max(self.rect.y, 0)
        self.rect.y = min(self.rect.y, screen_height - ground.get_height())

        self.vel = max(least_vel, self.vel - gravity)

        if not self.gameover:
            self.cnt = (self.cnt + 1) % flap_cooldown

        if self.cnt == 0:
            self.index = (self.index + 1) % self.num_sprites

        self.image = pygame.transform.rotate(self.images[self.index], self.vel * 4)

    def draw(self):
        screen.blit(self.image, self.rect)


class Pipe(pygame.sprite.Sprite):
    def __init__(self, x, y, top):
        super().__init__()
        self.image = pygame.image.load('images\\pipe.png')
        self.image = pygame.transform.scale(self.image, (self.image.get_width(), self.image.get_height() * 1.5))

        self.rect = self.image.get_rect()

        if top:
            self.image = pygame.transform.flip(self.image, False, True)
            self.rect.bottomleft = [x, y]
        else:
            self.rect.topleft = [x, y]

    def update(self):
        self.rect.x -= ground_speed

    def draw(self, surface):
        surface.blit(self.image, self.rect)


birds = pygame.sprite.Group()

birds_list = [Bird(bird_start_x, screen_height / 2) for _ in range(num_of_birds)]


for bird in birds_list:
    birds.add(bird)

pipes = pygame.sprite.Group()


def generate_next_gen():
    sorted_birds = sorted(birds_list, key=lambda bird: bird.life_time)[-10:]

    for i, best_bird in enumerate(sorted_birds):
        next_ten_birds = birds_list[i:i + 10]
        for bird in next_ten_birds:
            bird.brain = copy.deepcopy(best_bird.brain)
            bird.brain.randomize()


def reset_game():

    generate_next_gen()

    for bird in birds:
        bird.brain.train(bird.all_states)

    pipes_to_pass.clear()
    for bird in birds_list:
        bird.life_time += 1
        bird.rect.y = screen_height / 2
    pipes.empty()

    for bird in birds_list:
        bird.all_states = []
        bird.rect.x = bird_start_x
        bird.gameover = False


while run:

    clock.tick(60)

    screen.blit(background, (0, 0))

    all_game_overs = 0

    for bird in birds_list:
        if bird.gameover:
            all_game_overs += 1

    if all_game_overs == len(birds_list):
        game_over = True

    if not game_over:

        ground_pos -= ground_speed

        if ground_pos <= -80:
            ground_pos = 0

    if not game_over:

        now = pygame.time.get_ticks()

        if now - last_pipe > pipe_freq:

            middle = screen_height // 2 + random.randint(-screen_height // 4, screen_height // 4)
            gap = screen_height / 7

            pb = Pipe(screen_width, middle + gap // 2, False)
            pt = Pipe(screen_width, middle - gap // 2, True)

            pipes.add(pb)
            pipes.add(pt)

            pipes_to_pass.append(pb)
            pipes_to_pass.append(pt)

            last_pipe = now

    birds.update()
    birds.draw(screen)

    if not game_over:
        pipes.update()

    pipes.draw(screen)

    max_x = max([bird.rect.x for bird in birds_list])

    if len(pipes_to_pass) > 0:

        x_cord, _ = pipes_to_pass[0].rect.topright

        if max_x >= x_cord:
            score += 1
            pipes_to_pass = pipes_to_pass[2:]

    display_text(str(score), 72, (255, 255, 255), screen_width / 2, 50)
    display_text("Generation: " + str(generation), 36, (0, 0, 0), screen_width / 20, 70)

    screen.blit(ground, (ground_pos, screen_height - ground.get_height()))

    if game_over:
        if pygame.key.get_pressed()[K_SPACE] or True:
            game_over = False
            score = 0
            pipe_freq = 2000
            last_pipe = -pipe_freq
            generation += 1
            reset_game()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    pygame.display.update()

pygame.quit()

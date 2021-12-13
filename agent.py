import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNetwork, DeepQTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.002 # learning rate

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilion = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft if exceed the length
        self.model = Linear_QNetwork(11, 256, 3)  # input 11, output 3
        self.trainer = DeepQTrainer(self.model, lr=LR, gamma=self.gamma)
        # TODO: model, trainer--do optimization


    # caculate the current state of the game
    def get_state(self, game):
        head = game.snake[0]
        # BLOCK_SIZE 20
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # state: 11 boolean values:
        # Danger: Danger straight, Danger right, Danger left
        # Move direction: dir_l, dir_r, dir_u, dir_d,
        # Food location: food left, right, up, down
        # e.g. 1,0,0,0,0,1,0,0,1,1,0
        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done): # store everthing we need
        # done: game over
        # store one tuple
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEM is reached

    def train_long_memory(self):
        # grab a batch from memory
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        # train for one step
        self.trainer.train_step(state, action, reward, next_state, done)

    # Call model for getting the next state of the snake
    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        # logic: the more games we have, the smaller the epsilion is
        #        then the less possibility to get a random move
        self.epsilion = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilion: # random move
            move = random.randint(0, 2)
            final_move[move] = 1
        else: # move based on model
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)  # execute the forward function to predict
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    # training loop
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, replay the game, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record: # a new high score
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()

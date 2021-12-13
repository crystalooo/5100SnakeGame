import sys

#mathematical imports
import numpy as np
import math
import random

#openai imports
import gym
import gym_maze #this import requires python setup files


def simulate():

    #Intialize learning parameters to initial time frame (0th second)
    learningRate = findLearningRate(0)
    exploreRate = findExploreRate(0)
    #How much the RL agent cares about future rewards wrt current reward
    discountFactor = 0.99

    #Environment rendering + keeping track of streaks
    streaks = 0
    env.render()

    for episode in range(NUM_EPISODES):
        obv = env.reset()
        #Initial state
        state_0 = bucketState(obv)
        finalReward = 0

        for t in range(MAX_T):
            action = select_action(state_0, exploreRate)

            #Take next step
            obv, reward, done, _ = env.step(action)

            #Bucket result and update award
            state = bucketState(obv)
            finalReward += reward

            #Update QTable
            best_q = np.amax(q_table[state])
            q_table[state_0 + (action,)] += learningRate * (reward + discountFactor * (best_q) - q_table[state_0 + (action,)])

            #Updating state for next round
            state_0 = state

            # Print data
            if DEBUG_MODE == 2:
                print("\nEpisode = %d" % episode)
                print("t = %d" % t)
                print("Action: %d" % action)
                print("State: %s" % str(state))
                print("Reward: %f" % reward)
                print("Best Q: %f" % best_q)
                print("Explore rate: %f" % exploreRate)
                print("Learning rate: %f" % learningRate)
                print("Streaks: %d" % streaks)
                print("")

            elif DEBUG_MODE == 1:
                if done or t >= MAX_T - 1:
                    print("\nEpisode = %d" % episode)
                    print("t = %d" % t)
                    print("Explore rate: %f" % exploreRate)
                    print("Learning rate: %f" % learningRate)
                    print("Streaks: %d" % streaks)
                    print("Total reward: %f" % finalReward)
                    print("")

            if RENDER_MAZE:
                env.render()

            if env.is_game_over():
                sys.exit()

            if done:
                print("Episode %d finished after %f time steps with total reward = %f (streak %d)."
                      % (episode, t, finalReward, streaks))

                if t <= SOLVED_T:
                    streaks += 1
                else:
                    streaks = 0
                break

            elif t >= MAX_T - 1:
                print("Episode %d timed out at %d with total reward = %f."
                      % (episode, t, finalReward))

        #
        if streaks > STREAK_TO_END:
            break

        # Update parameters
        exploreRate = findExploreRate(episode)
        learningRate = findLearningRate(episode)


def select_action(state, exploreRate):
    #This is random exploration + exploitation step
    #Select random action
    if random.random() < exploreRate:
        action = env.action_space.sample()
    #Select action with highest q value
    else:
        action = int(np.argmax(q_table[state]))
    return action


def findExploreRate(t):
    return max(MINIMUM_ERATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))


def findLearningRate(t):
    return max(MINIMUM_LRATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))


def bucketState(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the buckets
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/bound_width
            scaling = (NUM_BUCKETS[i]-1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)


if __name__ == "__main__":

    #Using OpenAi Gym to make a maze environment with 10x10 cells
    env = gym.make("maze-random-10x10-plus-v0")

    # Number of discrete states
    MAZE_SIZE = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    NUM_BUCKETS = MAZE_SIZE  # one bucket per grid

    # Number of discrete actions
    NUM_ACTIONS = env.action_space.n  # ["N", "S", "E", "W"]
    # Bounds for each discrete state
    STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))

    #Learning constants
    MINIMUM_ERATE = 0.001
    MINIMUM_LRATE = 0.2
    DECAY_FACTOR = np.prod(MAZE_SIZE, dtype=float) / 10.0

    #Episode constants
    NUM_EPISODES = 50000
    MAX_T = np.prod(MAZE_SIZE, dtype=int) * 100
    STREAK_TO_END = 100 #Considered completed when beyond 120
    SOLVED_T = np.prod(MAZE_SIZE, dtype=int)
    DEBUG_MODE = 0
    RENDER_MAZE = True

    #QTable creation and initialization
    q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,), dtype=float)

    simulate()


# SnakeGameAI - Group2 Project
> An attempt at teaching a Snake agent to learn how to beat the famous Snake Game, and hopefully draw insights from its learning to apply to real-life scenarios.

## Project Description

### Background:

This is the course project for CS5100 Foundation of AI.

### Contributors:

This project exists thanks to all the people who contribute:  
Xiaoli Ou,
Aparna Krishnan.

### Purpose:

Through this project, we seek to establish this link between Game AI and real-life navigation, and draw useful insights in making agent-based navigation more intuitive and rational.

## High-Level Design
### Our Solution
Reinforcement learning (RL) is an area of machine learning concerned with how software agents ought to take actions in an environment in order to maximize the notion of cumulative reward. We will use deep Q learning to extend reinforcement learning by using a deep neural network to predict the actions.
![Link](https://github.com/crystalooo/5100SnakeGame/blob/main/components.jpg)
### Three Components: Game, Agent, Model
We have broken down the project implementation into three components: Environment(Game), Agent, and Model(Neural Network). (Linking of Components is shown in the flowing pictures).

![Design](https://github.com/crystalooo/5100SnakeGame/blob/main/ThreeParts.png)

### Neural Network Model
We are using a neural network with an input layer of size 11 and one hidden layer with 256 neurons and an output layer of 3 neurons. The neural network maps input states to (action, Q-value) pairs. In this case, each output node (representing an action) contains the action’s q-value as a floating point number.
![Model](https://github.com/crystalooo/5100SnakeGame/blob/main/model.png)

## Deep Q Learning Steps
1. The game starts, and the Q-value is randomly initialized(init model)
2. The system gets the current state s.
3. Based on s, it chooses action,  randomly or based on its neural network. During the first phase of the training, the system often chooses random actions to maximize exploration. Later on, the system relies more and more on its neural network  to predict the action.(tradeoff between Exploration and Exploitation)
4. When the snake performs the chosen action, the environment gives a reward.
5. The agent reaches the new state and updates its Q-value. Q Update Rule: Bellman equation.
6. Also, for each move, it stores the original state, the action, the state reached after performing that action, the reward obtained and whether the game ended or not. This data is later sampled to train the neural network. 
7. Repeat step2+step3+step4+step5+step6


## Usage

- Use a separate environment and install all the required modules using anaconda.

## Referrence
https://towardsdatascience.com/deep-q-learning-tutorial-mindqn-2a4c855abffc
https://gym.openai.com/
https://github.com/MattChanTK/gym-maze/blob/master/README.md
https://www.geeksforgeeks.org/snake-game-in-python-using-pygame-module/
https://www.youtube.com/playlist?list=PLqnslRFeH2UrDh7vUmJ60YrmWd64mTTKV


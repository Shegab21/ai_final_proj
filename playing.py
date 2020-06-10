import random
import numpy as np
import gym

class Agent:
    def __init__(self, env):
        self.action_size = env.action_space.n
        print("Action Size :", self.action_size)

    def get_action(self, state):
        action = random.choice(range(self.action_size))
        pole_angle = state[2]
        action = 0 if pole_angle < 0 else 1
        return action

class Agent2:
    def __init__(self, env):
        self.is_discrete = type(env.action_space) == gym.spaces.discrete.Discrete
        if self.is_discrete:
            self.action_size = env.action_space.n
            print("Action Size :", self.action_size)
        else:
            self.action_low = env.action_space.low
            self.action_high = env.action_space.high
            self.action_shape = env.action_space.shape
            print("Action range: ", self.action_low, self.action_high)

    def get_action(self, state):
        if self.is_discrete:
            action = random.choice(range(self.action_size))
        else:
            action = np.random.uniform(self.action_low, self.action_high, self.action_shape)
        pole_angle = state[2]
        action = 0 if pole_angle < 0 else 1
        return action

def trying_1():
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    agent = Agent2(env)
    state = env.reset()
    for i in range(200):
        # action = env.action_space.sample()
        action = agent.get_action(state)
        state, reward, done, info = env.step(action)
        env.render()


def trying_2():
    env_name = "FrozenLake-v0"
    env = gym.make(env_name)
    agent = Agent(env)
    state = env.reset()
    for i in range(200):
        # action = env.action_space.sample()
        action = agent.get_action(state)
        state, reward, done, info = env.step(action)
        env.render()

def main():
    trying_1()


if __name__ == '__main__':
    main()
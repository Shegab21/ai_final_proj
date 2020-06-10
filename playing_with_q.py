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

def trying_1():
    env_name = "CartPole-v1"
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
import random
import numpy as np
import gym
import time
from gym.envs.registration import register
from IPython.display import clear_output

class Agent:
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
        return action

class QAgent(Agent):
    def __init__(self, env, discount_rate=0.97, learning_rate=0.01):
        super().__init__(env)
        self.state_size = env.observation_space.n
        print("State size:", self.state_size)

        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.epsilon = 1.0
        self.build_model()

    def build_model(self):
        self.q_table = 1e-4*np.random.random([self.state_size, self.action_size])

    def get_action(self, state):
        q_state = self.q_table[state]
        action_greedy = np.argmax(q_state)
        action_random = super().get_action(state)
        return action_random if random.random() < self.epsilon else action_greedy

    def train(self, experience):
        state, action, next_state, reward, done = experience
        q_next = self.q_table[next_state]
        q_next = np.zeros([self.action_size]) if done else q_next
        q_target = reward + self.discount_rate + np.max(q_next)

        q_update = q_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * q_update

        if done:
            self.epsilon = self.epsilon * .99


def trying():
    try:
        register(
            id="FrozenLakeNoSlip-v0",
            entry_point="gym.envs.toy_text:FrozenLakeEnv",
            kwargs={"map_name": "4x4", "is_slippery": False},
            max_episode_steps=100,
            reward_threshold=0.78  # optimum = .8196
        )

    except:
        pass

    env_name = "FrozenLake-v0"
    env = gym.make(env_name)
    agent = Agent(env)
    state = env.reset()
    done = False
    while not done:
        # action = env.action_space.sample()
        action = agent.get_action(state)
        state, reward, done, info = env.step(action)
        print("s:", state, "a:", action)
        env.render()
        time.sleep(.5)
        clear_output(wait=True)


def trying2():
    try:
        register(
            id="FrozenLakeNoSlip-v0",
            entry_point="gym.envs.toy_text:FrozenLakeEnv",
            kwargs={"map_name": "4x4", "is_slippery": False},
            max_episode_steps=100,
            reward_threshold=0.78  # optimum = .8196
        )

    except:
        pass

    env_name = "FrozenLake-v0"
    env = gym.make(env_name)

    agent = QAgent(env)

    total_reward = 0
    for episode in range(100):
        state = env.reset()
        done = False
        while not done:
            # action = env.action_space.sample()
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            agent.train((state, action, next_state, reward, done))
            state = next_state
            total_reward += reward
            print("s:", state, "a:", action)
            print("Episode: {}, Total reward: {}, Epsilon: {}".format(episode, reward, agent.epsilon))
            env.render()
            time.sleep(.05)
            clear_output(wait=True)

def main():
    trying2()
    trying2()


if __name__ == '__main__':
    main()

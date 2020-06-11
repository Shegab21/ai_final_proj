import ast
import random
import numpy as np
import gym
import time
from gym.envs.registration import register
from IPython.display import clear_output
import os


class Agent:
    def __init__(self, env):
        self.action_size = env.action_space.n
        print("Action Size :", self.action_size)

    def get_action(self, state):
        action = random.choice(range(self.action_size))
        return action


class QAgent(Agent):
    def __init__(self, env, discount_rate=0.97, learning_rate=0.01):
        super().__init__(env)
        #self.state_size = env.observation_space.low
        self.state_size = [(83.6, 4.8, -200, 200)]
        print("Type:", type(env.action_space))
        print("State size:", self.state_size)

        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.epsilon = 1.0
        self.build_model()

    def build_model(self):
        f = open("q_table.txt", "rw+")
        if os.path.getsize("q_table.txt") > 0:
            file_str = f.read()
            self.q_dict = ast.literal_eval(file_str)
        else:
            self.q_dict = {}
        f.close()
        # self.q_table = 1e-4 * np.random.random([self.state_size, self.action_size])
        # self.q_table = np.zeros((int(83.6) * int(4.8) * 200 * 200 * 10000, self.action_size))


    def get_action(self, state):
        # q_state = self.q_table[state]
        state_tup = tuple((round(state[0], 1), round(state[1], 1),
                           round(state[2], 1), round(state[3], 1)))
        q_state = self.q_dict.get(state_tup, [0, 0])
        if q_state == [0, 0]:
            self.q_dict[state_tup] = [0, 0]
        action_greedy = np.argmax(q_state)
        action_random = super().get_action(state)
        return action_random if random.random() < self.epsilon else action_greedy

    def train(self, experience):
        state, action, next_state, reward, done = experience
        # q_next = self.q_table[next_state]
        # q_next = self.q_dict.get((next_state, action), 0)
        # if q_next == 0:
        #     self.q_dict[(next_state, action)] = 0
        #
        # self.q_dict[(state, action)] =  reward + self.discount_rate
        # best_move = np.max(q_next)
        state_tup = tuple((round(state[0], 1), round(state[1], 1),
                                  round(state[2], 1), round(state[3], 1)))
        q_next = self.q_dict.get(tuple(next_state), [0, 0])
        if q_next == [0, 0]:
            self.q_dict[tuple(next_state)] = [0, 0]
        old_left, old_right = self.q_dict[state_tup]
        if action == 0:
            q_target = np.max(q_next) * self.discount_rate + reward
            q_updated = (q_target - self.q_dict[state_tup][0]) * self.learning_rate
            self.q_dict[state_tup] = (old_left + q_updated, old_right)
        else:
            q_target = np.max(q_next) * self.discount_rate + reward
            q_updated = (q_target - self.q_dict[state_tup][1]) * self.learning_rate
            self.q_dict[state_tup] = (old_left, old_right + q_updated)


        # q_next = np.zeros([self.action_size]) if done else q_next
        # q_target = reward + self.discount_rate * np.max(q_next)
        #
        # q_update = q_target - self.q_table[state, action]
        # self.q_table[state, action] += self.learning_rate * q_update

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
    # try:
    #     register(
    #         id="CartPole-v1",
    #         entry_point="gym.envs.toy_text:FrozenLakeEnv",
    #         kwargs={"map_name": "4x4", "is_slippery": False},
    #         max_episode_steps=100,
    #         reward_threshold=0.78  # optimum = .8196
    #     )
    #
    # except:
    #     pass

    env_name = "CartPole-v1"
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
            rounded_next_state = (round(next_state[0], 1), round(next_state[1], 1),
                                  round(next_state[2], 1), round(next_state[3], 1))
            agent.train((state, action, rounded_next_state, reward, done))
            state = next_state
            total_reward += reward
            # print("s:", state, "a:", action)
            # print("Episode: {}, Total reward: {}, Epsilon: {}".format(episode, total_reward, agent.epsilon))
            # print(agent.q_dict)
            env.render()
            # time.sleep(.05)
            # clear_output(wait=True)
    f = open("q_table.txt", "rw+")
    f.write(agent.q_dict)
    f.close()


def main():
    trying2()


if __name__ == '__main__':
    main()

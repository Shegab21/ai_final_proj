import random
import numpy as np
import gym


class Agent:
    def __init__(self, env):
        self.action_size = env.action_space.n

    def get_action(self, state):
        action = random.choice(range(self.action_size))
        return action


class QAgent(Agent):
    def __init__(self, env, discount_rate=0.97, learning_rate=0.01):
        super().__init__(env)
        self.state_size = [(83.6, 4.8, -200, 200)]

        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.epsilon = 1.0
        self.q_dict = {}

    def get_action(self, state):
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

        if done:
            self.epsilon = self.epsilon * .99


def run():
    env_name = "CartPole-v1"
    env = gym.make(env_name)

    agent = QAgent(env)

    total_reward = 0
    for episode in range(100):
        state = env.reset()
        done = False
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            rounded_next_state = (round(next_state[0], 1), round(next_state[1], 1),
                                  round(next_state[2], 1), round(next_state[3], 1))
            agent.train((state, action, rounded_next_state, reward, done))
            state = next_state
            total_reward += reward
            env.render()


def main():
    run()


if __name__ == '__main__':
    main()

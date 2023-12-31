# Reinforcement Learning atari taxi-v3

import gymnasium as gym
import random
import numpy as np
import time
import matplotlib.pyplot as plt

# sample action
def sample_action(obs, epsilon):
    rv = random.random()
    if rv < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[obs])

# Q-learning 학습
def train():
    alpha = 0.1
    epsilon = 0.1
    gamma = 0.6

    for n_epi in range(100000):
        s, info = env.reset()

        done = False
        score = 0.0

        while not done:
            a = sample_action(s, epsilon)
            s_prime, r, terminated, truncated, info = env.step(a)
            
            q_value = q_table[s, a]
            max_q_prime = np.max(q_table[s_prime])

            #q-learning update rule
            q_table[s, a] = q_value + alpha * (r + gamma*max_q_prime - q_value)
            
            s = s_prime
            score += r
            
            done = (terminated or truncated)
        if (n_epi != 0) and (n_epi % 20 == 0):
            print("n_epi: {}, score : {:.1f}, eps: {:.1f}%".format(n_epi, score, epsilon*100))
            
# evaluation            
def eval():
    for n_epi in range(10):
        s, info = env_eval.reset()

        done = False
        score = 0
        iter = 0

        while not done:
            a = np.argmax(q_table[s])
            s, r, terminated, truncated, _ = env_eval.step(a)

            score += r
            iter += 1
            done = (terminated or truncated)
            # logs in episode
            print("action: {}, reward : {:.1f}".format(a, r))
        # logs per episode
        print("n_epi: {}, score : {:.1f}, iter : {}".format(n_epi, score, iter))

#main
if __name__ == '__main__':
    env = gym.make("Taxi-v3")
    env_eval = gym.make("Taxi-v3", render_mode = "human")
    q_table = np.zeros([env.observation_space.n, env.action_space.n])    

    train()
    eval()

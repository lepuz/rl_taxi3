import gymnasium as gym
from gymnasium.envs.toy_text.taxi import *
import torch
from train import DQN
import numpy as np
import random

def load_model():

    # 딕셔너리 형태 파라미터 load
    checkpoint = torch.load('ckpt/checkpoint_ep_10_score_-1244.pt')
    # 모델 구조 생성
    model = DQN()
    # 모델 파라미터 load
    model.load_state_dict(checkpoint['model']) 
    return model
    

env = gym.make("Taxi-v3", render_mode = "human")
x = TaxiEnv()
state, info = env.reset()

model = load_model()
model.eval()


epsilon = 0.0

for n_epi in range(10):
    s, info = env.reset()

    done = False
    score = 0
    iter = 0

    while not done:
        taxi_row, taxi_col, passenger_idx, destination_idx = x.decode(s)
        state = torch.tensor(np.reshape([taxi_row, taxi_col, passenger_idx, destination_idx], [1, 4])).type(torch.float32)
        prediction = model(state)

        action = torch.argmax(prediction)
        action = prediction.argmax()

        rv = random.random()
        if rv < epsilon:
            action =  random.randint(0, 5)
        else:
            action = action.argmax().item()

        next_state, reward, terminated, truncated, info = env.step(action)

        print("n_epi: {}, action: {}, reward : {:.1f}".format(n_epi, action, reward))

        if terminated or truncated:
            state, info = env.reset()
        state = next_state

    print("n_epi: {}, score : {:.1f}, iter : {}".format(n_epi, score, iter))




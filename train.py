import gymnasium as gym
from gymnasium.envs.toy_text.taxi import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import collections
import numpy as np
from torch.utils.tensorboard import SummaryWriter

x = TaxiEnv()

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 6)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
    def sample_action(self, obs, epsilon):
        obs = decode_state(obs)
        out = self.forward(obs)
        rv = random.random()
        if rv < epsilon:
            return random.randint(0, 5)
        else:
            return out.argmax().item()

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=250000)
    
    def size(self):
        return len(self.buffer)
    
    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n_batch):
        mini_batch = random.sample(self.buffer, n_batch)

        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst), \
        torch.tensor(a_lst), \
        torch.tensor(r_lst), \
        torch.tensor(s_prime_lst), torch.tensor(done_mask_lst)
    
def decode_state(state, n_batch=1):
    if n_batch == 1 :
        taxi_row, taxi_col, passenger_idx, destination_idx = x.decode(state)
        s = torch.tensor(np.reshape([taxi_row, taxi_col, passenger_idx, destination_idx], [n_batch, 4])).type(torch.float32)
        return s
    else:
        lst = []
        for i in range(n_batch):
            taxi_row, taxi_col, passenger_idx, destination_idx = x.decode(state[i])
            lst.append([taxi_row, taxi_col, passenger_idx, destination_idx])
        s = torch.tensor(lst).type(torch.float32)
        #s = torch.tensor(np.reshape([taxi_row, taxi_col, passenger_idx, destination_idx], [n_batch, 4])).type(torch.float32)
        return s

def train(q, q_target, memory, optimizer):
    gamma=0.98
    n_batch = 32
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(n_batch)
        s = decode_state(s, n_batch)
        #s = torch.tensor(np.reshape([taxi_row, taxi_col, passenger_idx, destination_idx], [n_batch, 4])).type(torch.float32)

        q_out = q(s)

        #q_a = q_out.gather(1, a.to(device))
        #action에 대한 q값을 얻어오기 위함, predicted value
        q_a = q_out.gather(1, a)

        s_prime = decode_state(s_prime, n_batch)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)

        #target = r.to(device) + gamma * max_q_prime * done_mask.to(device)
        target = r + gamma * max_q_prime * done_mask
        loss = F.mse_loss(q_a, target)
        print(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main():
    #env = gym.make("Taxi-v3", render_mode = "human")
    env = gym.make("Taxi-v3")
    writer = SummaryWriter(log_dir='logs')


    #q = DQN().to(device)
    #q_target = DQN().to(device)
    q = DQN()
    q_target = DQN()
    #copy param from q to q_target
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()
    optimizer = optim.Adam(q.parameters(), lr=0.0005)

    min_eval_score = -9999

    for n_epi in range(10000000):
        epsilon = max(0.1, 0.5 - 0.001*(n_epi/200))

        s, info = env.reset()
        
        done = False
        score = 0.0

        while not done:
            #s = torch.tensor(np.reshape(s, [1, 1])).type(torch.float32)            

            #taxi_row, taxi_col, passenger_idx, destination_idx = x.decode(s)
            #s = torch.tensor(np.reshape([taxi_row, taxi_col, passenger_idx, destination_idx], [1, 4])).type(torch.float32)
            a = q.sample_action(s, epsilon)
            s_prime, r, terminated, truncated, info = env.step(a)

            done = (terminated or truncated)
            # done : done_mask set 0, not done : done_mask set 1
            done_mask = 0.0 if done else 1.0

            memory.put((s, a, r, s_prime, done_mask))
            
            s = s_prime
            score += r
            #print(score, r)
            if done: break

        if memory.size() > 2000:
            train(q, q_target, memory, optimizer)
            
            if score > min_eval_score:
                torch.save({'model': q.state_dict(), 'score':score, 'eps : ':epsilon*100},'ckpt/checkpoint_ep_%d_score_%d.pt'%(n_epi,score))
                min_eval_score = score
        writer.add_scalar("score/episode", score, n_epi)

        if (n_epi != 0) and (n_epi % 20 == 0):
            q_target.load_state_dict(q.state_dict())
            print("n_epi: {}, score : {:.1f}, n_buffer: {}, eps: {:.1f}%".format(n_epi, score, memory.size(), epsilon*100))
            score = 0.0
    writer.flush()
    writer.close()

if __name__ == '__main__':
    main()

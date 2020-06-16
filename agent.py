import numpy as np
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim




Experience = collections.namedtuple("Experience", field_names=["state", "action", "reward", "done", "new_state"])

class ExperienceBuffer:

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, np.float32), np.array(dones, np.bool), np.array(next_states)


class Net(nn.Module):

    def __init__(self, num_states, num_actions):
        super(Net, self).__init__()
        self.dense1 = nn.Linear(num_states, 256)
        self.dense2 = nn.Linear(256,256)
        # self.dense3 = nn.Linear(256, 512)
        # self.dense4 = nn.Linear(512,512)
        self.output = nn.Linear(256, num_actions)

    def forward(self, state):
        t=state
        t = F.relu(self.dense1(t))
        t = F.relu(self.dense2(t))
        # t = F.relu(self.dense3(t))
        # t = F.relu(self.dense4(t))
        t = self.output(t)
        
        return t



    


class Agent:
    
    def __init__(self, num_states, num_actions, eps, eps_decay, batch_size, discount, learning_rate, replay_buffer_cap):
        self.num_states = num_states
        self.num_actions = num_actions
        self.approximator = Net(num_states, num_actions)
        self.experience_buffer = ExperienceBuffer(replay_buffer_cap)
        self.eps = eps
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.discount = discount
        self.optimizer = optim.Adam(self.approximator.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def policy(self, state):
        '''
        Takes a state and returns Q values
        '''
        if np.random.random() < self.eps:
            return np.random.randint(0, self.num_actions)

        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            self.approximator.eval()
            with torch.no_grad():
                actions = self.approximator(state)
            return torch.argmax(actions).item()

    def update_q_function(self):
        
        self.approximator.train()
        batch_size = self.batch_size if len(self.experience_buffer)>self.batch_size else len(self.experience_buffer)
        
        states, actions, rewards, dones, next_states = self.experience_buffer.sample(batch_size)
        states, actions, rewards, dones, next_states = torch.FloatTensor(states), actions, torch.FloatTensor(rewards), torch.LongTensor(dones), torch.FloatTensor(next_states)
        predicted = self.approximator(states)
        with torch.no_grad():
            actual = predicted.detach().clone()
            ind = np.arange(batch_size,dtype = np.int8)
            #maxi = torch.max(self.approximator(next_states), dim=1)
            #maxi = maxi.values
            #mask = (1-dones)
            #next_vals = maxi*mask
            actual[ind,actions] = rewards + self.discount*((torch.max(self.approximator(next_states), dim=1)).values*(1-dones))
        self.optimizer.zero_grad()
        loss = torch.mean((actual - predicted)**2)
        #loss = self.criterion(predicted,actual)
        loss.backward()
        self.optimizer.step()
        self.eps = max(0.01,self.eps - self.eps_decay)
        return loss.item()

    def store_experience(self, state, action, reward, done, next_state):
        exp = Experience(state, action, reward, done, next_state)
        self.experience_buffer.append(exp)



    







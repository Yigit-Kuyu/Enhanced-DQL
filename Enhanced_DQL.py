import gym
import gym_maze
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import time
from torch.distributions import Categorical
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# dimensions
input_dim = 2  # x and y coordinates
action_dim = 4  # up, down, left, right
fc1_output_dim = 64
hidden_size = 128
fc2_output_dim = action_dim

batch_size = 32
sequence_length = 1
num_of_train_iter = 10000
num_train_episodes = 200
num_of_test_iter = 10000
num_test_episodes = 20 
gamma = 0.99
target_update = 10
memory_size = 10000

# Transformer-specific parameters 
num_heads = 4 # multi-head attention
num_layers = 2 # Transformer encoder layers


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerNetwork(nn.Module):
    def __init__(self, input_dim, hidden_size, num_heads, num_layers):
        super(TransformerNetwork, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, max_len=5000)
        encoder_layers = nn.TransformerEncoderLayer(hidden_size, num_heads, dim_feedforward=hidden_size*4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return x

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, fc1_output_dim) #  First fully connected layer
        self.transformer = TransformerNetwork(fc1_output_dim, hidden_size, num_heads, num_layers) # # Transformer layers
        self.fc2 = nn.Linear(hidden_size, fc2_output_dim)  # Final output layer
        self.relu = nn.ReLU()

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        elif x.dim() == 3:
            x = x.transpose(0, 1)  
         
        x = self.relu(self.fc1(x))
        x = self.transformer(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def select_action(self, state, epsilon):  # Stochastic policy
        with torch.no_grad():
            q_values = self.forward(state) 
            q_values = q_values.squeeze(0).squeeze(0)  
            
            # Convert Q-values to probabilities using softmax
            action_probs = F.softmax(q_values, dim=-1)
            
            # Create a categorical distribution and sample an action
            dist = Categorical(action_probs)
            action = dist.sample()
        
        return action.item()


class PrioritizedReplayMemory: # Prioritized experience replay for focusing on important transitions
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.eps = 1e-5  

    def push(self, state, action, next_state, reward, done):
        max_priority = np.max(self.priorities) if self.memory else 1.0
        
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        
        self.memory[self.position] = (state, action, next_state, reward, done)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.memory) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
        
        probabilities = (priorities + self.eps) / np.sum(priorities + self.eps)
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities, replace=False)
        samples = [self.memory[idx] for idx in indices]     
        return samples, indices

    def update_priorities(self, indices, priorities):
        self.priorities[indices] = priorities

    def __len__(self):
        return len(self.memory)

    def current_size(self):
        return min(self.position, self.capacity)


def optimize_model(policy_net, target_net, memory, optimizer): 
    if len(memory) < batch_size:
        return
    transitions, indices = memory.sample(batch_size)
    batch = list(zip(*transitions))

    state_batch = torch.cat(batch[0]).to(device)
    action_batch = torch.cat(batch[1]).to(device)
    next_state_batch = torch.cat(batch[2]).to(device)
    reward_batch = torch.cat(batch[3]).to(device).unsqueeze(1)  
    done_batch = torch.cat(batch[4]).to(device).unsqueeze(1)  

    # Get Q values for current state-action pairs
    q_values = policy_net(state_batch).squeeze(0)  
    state_action_values = q_values.gather(1, action_batch)  
    
    with torch.no_grad():
        next_q_values = target_net(next_state_batch).squeeze(0)  
        next_state_values = next_q_values.max(1)[0].unsqueeze(1)  
        expected_state_action_values = reward_batch + (gamma * next_state_values * (1 - done_batch)) 

    
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values, reduction='none')
    
    # Update priorities in the replay memory
    priorities = loss.detach().cpu().numpy() + 1e-5 
    priorities_flat = priorities.squeeze(1)
    memory.update_priorities(indices, priorities_flat)
    loss = loss.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



def train(env, policy_net, target_net, memory, optimizer, num_episodes, num_of_train_iter):
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay_steps = num_train_episodes * num_of_train_iter // 2 
    total_steps = 0
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor([state], device=device, dtype=torch.float32).unsqueeze(0)
        episode_reward = 0


        for t in range(num_of_train_iter):
            # Linear decay of epsilon for exploration-exploitation balance
            epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * total_steps / epsilon_decay_steps)
            action = policy_net.select_action(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            next_state = torch.tensor([next_state], device=device, dtype=torch.float32).unsqueeze(0)
            reward = torch.tensor([reward], device=device)
            done = torch.tensor([float(done)], device=device)

            memory.push(state, torch.tensor([[action]], device=device), next_state, reward, done)
            state = next_state
            episode_reward += reward.item()

            optimize_model(policy_net, target_net, memory, optimizer)
            total_steps += 1

            
            print(f"DQN Training episode: {episode}, step: {t}, action: {action}, reward: {reward.item()}")
            if done:
                torch.save(policy_net.state_dict(), 'E_dql.pkl')
                break


        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if episode % 10 == 0:
            print(f'DQN Training Episode:{episode}, episode reward is {episode_reward}')
            torch.save(policy_net.state_dict(), 'E_dql.pkl')

def test(env, policy_net, num_episodes, num_of_test_iter):
    total_rewards = []
    policy_net.load_state_dict(torch.load('E_dql.pkl'))
    for ep in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor([state], device=device, dtype=torch.float32).unsqueeze(0)
        episode_reward = 0
        done = False

        
        for iter in range(num_of_test_iter):
            action = policy_net.select_action(state, 0.05)  
            next_state, reward, done, _, _ = env.step(action)
            next_state = torch.tensor([next_state], device=device, dtype=torch.float32).unsqueeze(0)
            episode_reward += reward
            state = next_state
            print(f"DQN Testing episode: {ep}, step: {iter}, reward: {reward}, action: {action}")
            
            if done:
                break

        print(f'DQN testing Episode:{ep}, episode reward is {episode_reward}')
        total_rewards.append(episode_reward)
    
    
    return total_rewards

if __name__ == '__main__':
    env = gym.make("maze-sample-10x10-v0", apply_api_compatibility=True, render_mode="human")
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    memory = PrioritizedReplayMemory(memory_size)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-4)

    # Training phase
    train(env, policy_net, target_net, memory, optimizer, num_train_episodes, num_of_train_iter)

    # Testing phase
    start_time = time.time()
    test_rewards = test(env, policy_net, num_test_episodes, num_of_test_iter)
    end_time = time.time()

    total_time = end_time - start_time

    print('Average test reward:', np.mean(test_rewards))
    print(f"Testing time: {total_time:.2f} seconds")
    
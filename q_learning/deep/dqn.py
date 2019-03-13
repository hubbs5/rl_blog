#!/usr/bin/env python

# DQN for playing OpenAI Atari RAM. For full writeup, visit:
# https://www.datahubbs.com

import numpy as np
import sys
import os
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import gym
import torch
from torch import nn
from collections import namedtuple, deque, OrderedDict
from copy import copy, deepcopy
import pandas as pd
import time
import shutil

def main(argv):

    args = parse_arguments()
    if args.gpu is None or args.gpu == False:
        args.gpu = 'cpu'
    else:
        args.gpu = 'cuda'

    # Initialize environment
    env = gym.make(args.env)

    # Initialize DQNetwork
    dqn = QNetwork(env=env, 
        n_hidden_layers=args.hl, 
        n_hidden_nodes=args.hn, 
        learning_rate=args.lr, 
        bias=args.bias,
        tau=args.tau,
        device=args.gpu)
    # Initialize DQNAgent
    agent = DQNAgent(env, dqn,  
        memory_size=args.memorySize,
        burn_in=args.burnIn,
        reward_threshold=args.threshold,
        path=args.path) 
    print(agent.network)
    print(agent.target_network)
    # Train agent
    start_time = time.time()

    agent.train(epsilon=args.epsStart,
        gamma=args.gamma, 
        max_episodes=args.maxEps,
        batch_size=args.batch,
        update_freq = args.updateFreq,
        network_sync_frequency=args.netSyncFreq)
    end_time = time.time()
    # Save results
    if agent.success:
        agent.save_results(args)
        if args.plot:
            agent.plot_rewards()
    else:
        shutil.rmtree(agent.path)

    x = end_time - start_time
    hours, remainder = divmod(x, 3600)
    minutes, seconds = divmod(remainder, 60)
    print("Peak mean reward: {:.2f}".format(
        max(agent.mean_training_rewards)))
    print("Training Time: {:02}:{:02}:{:02}\n".format(
        int(hours), int(minutes), int(seconds)))


class DQNAgent:
    
    def __init__(self, env, network, memory_size=50000,
        batch_size=32, burn_in=10000, reward_threshold=None,
        path=None, *args, **kwargs):
        
        self.env = env
        self.env_name = env.spec.id
        self.network = network
        self.target_network = deepcopy(network)
        self.tau = network.tau
        self.batch_size = batch_size
        self.window = 100
        if reward_threshold is None:
            self.reward_threshold = 195 if 'CartPole' in self.env_name \
                else 300
        else:
            self.reward_threshold = reward_threshold
        self.path = path
        self.timestamp = time.strftime('%Y%m%d_%H%M')
        self.initialize(memory_size, burn_in)
           
    # Implement DQN training algorithm
    def train(self, epsilon=0.05, gamma=0.99, max_episodes=10000,
        batch_size=32, network_sync_frequency=5000, update_freq=4):
        self.gamma = gamma
        self.epsilon = epsilon
        # Populate replay buffer
        while self.buffer.burn_in_capacity() < 1:
            done = self.take_step(mode='explore')
            if done:
                self.s_0 = self.env.reset()
            
        ep = 0
        training = True
        while training:
            self.s_0 = self.env.reset()
            self.rewards = 0
            done = False
            while done == False:
                done = self.take_step(mode='train')
                # Update network
                if self.step_count % update_freq == 0:
                    self.update()
                # Sync networks
                if self.step_count % network_sync_frequency == 0:
                    self.target_network.load_state_dict(
                        self.network.state_dict())
                    
                if done:
                    ep += 1
                    self.training_rewards.append(self.rewards)
                    mean_rewards = np.mean(
                        self.training_rewards[-self.window:])
                    self.training_loss.append(np.mean(self.update_loss))
                    self.update_loss = []
                    self.mean_training_rewards.append(mean_rewards)
                    print("\rEpisode {:d} Mean Rewards {:.2f}\t\t".format(
                        ep, mean_rewards), end="")
                    
                    if ep >= max_episodes:
                        training = False
                        print('\nEpisode limit reached.')
                        break
                    if mean_rewards >= self.reward_threshold:
                        training = False
                        self.success = True
                        print('\nEnvironment solved in {} steps!'.format(
                            self.step_count))
                        break

    def take_step(self, mode='train'):
        if mode == 'explore':
            action = self.env.action_space.sample()
        else:
            s_0 = np.ravel(self.state_buffer)
            action = self.network.get_action(s_0, epsilon=self.epsilon)
            self.step_count += 1
        s_1, r, done, _ = self.env.step(action)
        self.rewards += r
        self.state_buffer.append(self.s_0.copy())
        self.next_state_buffer.append(s_1.copy())
        self.buffer.append(deepcopy(self.state_buffer), action, r, done, 
                           deepcopy(self.next_state_buffer))
        self.s_0 = s_1.copy()
        return done

    def calculate_loss(self, batch):
        states, actions, rewards, dones, next_states = [i for i in batch]
        rewards_t = torch.FloatTensor(rewards).to(device=self.network.device)
        actions_t = torch.LongTensor(np.array(actions)).to(
            device=self.network.device).reshape(-1,1)
        dones_t = torch.ByteTensor(dones).to(device=self.network.device)
        
        qvals = torch.gather(self.network.get_qvals(states), 1, 
            actions_t).squeeze()
        qvals_next = torch.max(self.target_network.get_qvals(next_states),
                               dim=-1)[0].detach()
        qvals_next[dones_t] = 0 # Zero-out terminal states
        expected_qvals = self.gamma * qvals_next + rewards_t
        loss = nn.MSELoss()(qvals, expected_qvals)
        return loss
    
    def update(self):
        self.network.optimizer.zero_grad()
        batch = self.buffer.sample_batch(batch_size=self.batch_size)
        loss = self.calculate_loss(batch)
        loss.backward()
        self.network.optimizer.step()
        if self.network.device == 'cuda':
            self.update_loss.append(loss.detach().cpu().numpy())
        else:
            self.update_loss.append(loss.detach().numpy())
        
    def initialize(self, memory_size, burn_in):
        self.buffer = experienceReplayBuffer(memory_size, burn_in)
        self.training_rewards = []
        self.training_loss = []
        self.update_loss = []
        self.mean_training_rewards = []
        self.rewards = 0
        self.step_count = 0
        self.s_0 = self.env.reset()
        self.state_buffer = deque(maxlen=self.tau)
        self.next_state_buffer = deque(maxlen=self.tau)
        [self.state_buffer.append(np.zeros(self.s_0.size)) 
         for i in range(self.tau)]
        [self.next_state_buffer.append(np.zeros(self.s_0.size)) 
         for i in range(self.tau)]
        self.state_buffer.append(self.s_0)
        self.success = False
        if self.path is None:
            self.path = os.path.join(os.getcwd(), 
                self.env_name, self.timestamp)
        os.makedirs(self.path, exist_ok=True)

    def plot_rewards(self):
        plt.figure(figsize=(12,8))
        plt.plot(self.training_rewards, label='Rewards')
        plt.plot(self.mean_training_rewards, label='Mean Rewards')
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.ylim([0, np.round(self.reward_threshold)*1.05])
        plt.savefig(os.path.join(self.path, 'rewards.png'))
        plt.show()

    def save_results(self, args):
        weights_path = os.path.join(self.path, 'dqn_weights.pt')
        torch.save(self.network.state_dict(), weights_path)
        # Save rewards
        rewards = pd.DataFrame(self.training_rewards, columns=['reward'])
        rewards.insert(0, 'episode', rewards.index.values)
        rewards.to_csv(os.path.join(self.path, 'rewards.txt'))
        # Save model parameters
        file = open(os.path.join(self.path, 'parameters.txt'), 'w')
        file.writelines('rewards')
        [file.writelines('\n' + str(k) + ',' + str(v)) 
            for k, v in vars(args).items()]
        file.close()

class QNetwork(nn.Module):
    
    def __init__(self, env, learning_rate=1e-3, n_hidden_layers=4,
        n_hidden_nodes=256, bias=True, activation_function='relu',
        tau=1, device='cpu', *args, **kwargs):
        super(QNetwork, self).__init__()
        self.device = device
        self.actions = np.arange(env.action_space.n)
        self.tau = tau
        n_inputs = env.observation_space.shape[0] * tau
        self.n_inputs = n_inputs
        n_outputs = env.action_space.n

        activation_function = activation_function.lower()
        if activation_function == 'relu':
            act_func = nn.ReLU()
        elif activation_function == 'tanh':
            act_func = nn.Tanh()
        elif activation_function == 'elu':
            act_func = nn.ELU()
        elif activation_function == 'sigmoid':
            act_func = nn.Sigmoid()
        elif activation_function == 'selu':
            act_func = nn.SELU()
        
        # Build a network dependent on the hidden layer and node parameters
        layers = OrderedDict()
        n_layers = 2 * (n_hidden_layers-1)
        for i in range(n_layers + 1):
            if n_hidden_layers == 0:
                layers[str(i)] = nn.Linear(
                    n_inputs,
                    n_outputs,
                    bias=bias)
            elif i == n_layers:
                layers[str(i)] = nn.Linear(
                    n_hidden_nodes,
                    n_outputs,
                    bias=bias)
            elif i % 2 == 0 and i == 0:
                layers[str(i)] = nn.Linear(
                    n_inputs,
                    n_hidden_nodes,
                    bias=bias)
            elif i % 2 == 0 and i < n_layers - 1:
                layers[str(i)] = nn.Linear(
                    n_hidden_nodes,
                    n_hidden_nodes,
                    bias=bias)
            else:
                layers[str(i)] = act_func
                
        self.network = nn.Sequential(layers)
        
        # Set device for GPU's
        if self.device == 'cuda':
            self.network.cuda()
        
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=learning_rate)
                
    def get_action(self, state, epsilon=0.05):
        if np.random.random() < epsilon:
            action = np.random.choice(self.actions)
        else:
            action = self.greedy_action(state)
        return action
    
    def greedy_action(self, state):
        qvals = self.get_qvals(state)
        return torch.max(qvals, dim=-1)[1].item()
    
    def get_qvals(self, state):
        if type(state) is tuple:
            state = np.array([np.ravel(s) for s in state])
        state_t = torch.FloatTensor(state).to(device=self.device)
        return self.network(state_t)

class experienceReplayBuffer:

    def __init__(self, memory_size=50000, burn_in=10000):
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.Buffer = namedtuple('Buffer', 
            field_names=['state', 'action', 'reward', 'done', 'next_state'])
        self.replay_memory = deque(maxlen=memory_size)

    def sample_batch(self, batch_size=32):
        samples = np.random.choice(len(self.replay_memory), batch_size, 
                                   replace=False)
        # Use asterisk operator to unpack deque 
        batch = zip(*[self.replay_memory[i] for i in samples])
        return batch

    def append(self, state, action, reward, done, next_state):
        self.replay_memory.append(
            self.Buffer(state, action, reward, done, next_state))

    def burn_in_capacity(self):
        return len(self.replay_memory) / self.burn_in
    
    def capacity(self):
        return len(self.replay_memory) / self.memory_size

def parse_arguments():
    parser = ArgumentParser(description='Deep Q Network Argument Parser')
    # Network parameters
    parser.add_argument('--hl', type=int, default=3,
        help='An integer number that defines the number of hidden layers.')
    parser.add_argument('--hn', type=int, default=16,
        help='An integer number that defines the number of hidden nodes.')
    parser.add_argument('--lr', type=float, default=0.001,
        help='An integer number that defines the number of hidden layers.')
    parser.add_argument('--bias', type=str2bool, default=True,
        help='Boolean to determine whether or not to use biases in network.')
    parser.add_argument('--actFunc', type=str, default='relu',
        help='Set activation function.')
    parser.add_argument('--gpu', type=str2bool, default=False,
        help='Boolean to enable GPU computation. Default set to False.')
    # Environment
    parser.add_argument('--env', dest='env', type=str, default='CartPole-v0')

    # Training parameters
    parser.add_argument('--gamma', type=float, default=0.99,
        help='A value between 0 and 1 to discount future rewards.')
    parser.add_argument('--maxEps', type=int, default=10000,
        help='An integer number of episodes to train the agent on.')
    parser.add_argument('--netSyncFreq', type=int, default=2000,
        help='An integer number that defines steps to update the target network.')
    parser.add_argument('--updateFreq', type=int, default=1,
        help='Integer value that determines how many steps or episodes' + 
        'must be completed before a backpropogation update is taken.')
    parser.add_argument('--batch', type=int, default=32,
        help='An integer number that defines the batch size.')
    parser.add_argument('--memorySize', type=int, default=50000,
        help='An integer number that defines the replay buffer size.')
    parser.add_argument('--burnIn', type=int, default=20000,
        help='Set the number of random burn-in transitions before training.')
    parser.add_argument('--epsStart', type=float, default=0.05,
        help='Float value for the start of the epsilon decay.')
    parser.add_argument('--epsEnd', type=float, default=0.01,
        help='Float value for the end of the epsilon decay.')
    parser.add_argument('--epsStrategy', type=str, default='constant',
        help="Enter 'constant' to set epsilon to a constant value or 'decay'" + 
        "to have the value decay over time. If 'decay', ensure proper" + 
        "start and end values.")
    parser.add_argument('--tau', type=int, default=1,
    	help='Number of states to link together.')
    parser.add_argument('--epsConstant', type=float, default=0.05,
        help='Float to be used in conjunction with a constant epsilon strategy.')
    parser.add_argument('--window', type=int, default=100,
        help='Integer value to set the moving average window.')
    parser.add_argument('--plot', type=str2bool, default=True,
        help='If true, plot training results.')
    parser.add_argument('--path', type=str, default=None,
        help='Specify path to save results.')
    parser.add_argument('--threshold', type=int, default=195,
        help='Set target reward threshold for the solved environment.')
    args = parser.parse_args()

    return parser.parse_args()

def str2bool(argument):
    if argument.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif argument.lower() in ('no', 'false', 'f', 'n', '0'):
        return False 
    else:
        raise ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    main(sys.argv)

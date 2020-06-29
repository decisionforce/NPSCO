import os
import gym
import argparse
import matplotlib.pyplot as plt
from IPython import display

parser = argparse.ArgumentParser()
parser.add_argument(
    '--hid_num',
    type=int,
    default=256,
    help='number of hidden unit to use')
parser.add_argument(
    '--drop_prob',
    type=float,
    default=0.0,
    help='probability of dropout')
parser.add_argument(
    '--env_name',
    type=str,
    default=None,
    help='name of environment')
parser.add_argument(
    '--num_episode',
    type=int,
    default=1000,
    help='number of training episodes')
parser.add_argument(
    '--num_repeat',
    type=int,
    default=10,
    help='repeat the experiment for several times')
parser.add_argument(
    '--use_gpu',
    type=int,
    default=0)
config = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = str(config.use_gpu)
    
    
    
ENV_NAME = config.env_name
USE_GPU = True if 0<=config.use_gpu<=7 else False
env = gym.make(ENV_NAME)
env.reset()



rwds_history = []
for repeat in range(config.num_repeat):
    """
    Implementation of PPO
    ref: Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
    ref: https://github.com/Jiankai-Sun/Proximal-Policy-Optimization-in-Pytorch/blob/master/ppo.py
    ref: https://github.com/openai/baselines/tree/master/baselines/ppo2
    NOTICE:
        `Tensor2` means 2D-Tensor (num_samples, num_dims) 
    """

    import gym
    import torch
    import torch.nn as nn
    import torch.optim as opt
    from torch import Tensor
    from torch.autograd import Variable
    from collections import namedtuple
    from itertools import count
    import torch.nn.functional as F
    #import matplotlib
    #matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from os.path import join as joindir
    from os import makedirs as mkdir
    import pandas as pd
    import numpy as np
    import argparse
    import datetime
    import math
    import random


    Transition = namedtuple('Transition', ('state', 'value', 'action', 'logproba', 'mask', 'next_state', 'reward'))
    EPS = 1e-10
    RESULT_DIR = 'Result_PPO'
    mkdir(RESULT_DIR, exist_ok=True)
    mkdir(ENV_NAME.split('-')[0]+'/CheckPoints',exist_ok=True)
    mkdir(ENV_NAME.split('-')[0]+'/Rwds',exist_ok=True)
    rwds = []

    class args(object):
        env_name = ENV_NAME
        seed = 1234
        num_episode = config.num_episode
        batch_size = 2048
        max_step_per_round = 2000
        gamma = 0.995
        lamda = 0.97
        log_num_episode = 1
        num_epoch = 10
        minibatch_size = 256
        clip = 0.2
        loss_coeff_value = 0.5
        loss_coeff_entropy = 0.01
        lr = 3e-4
        num_parallel_run = 1
        # tricks
        schedule_adam = 'linear'
        schedule_clip = 'linear'
        layer_norm = True
        state_norm = False
        advantage_norm = True
        lossvalue_norm = True


    class RunningStat(object):
        def __init__(self, shape):
            self._n = 0
            self._M = np.zeros(shape)
            self._S = np.zeros(shape)

        def push(self, x):
            x = np.asarray(x)
            assert x.shape == self._M.shape
            self._n += 1
            if self._n == 1:
                self._M[...] = x
            else:
                oldM = self._M.copy()
                self._M[...] = oldM + (x - oldM) / self._n
                self._S[...] = self._S + (x - oldM) * (x - self._M)

        @property
        def n(self):
            return self._n

        @property
        def mean(self):
            return self._M

        @property
        def var(self):
            return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

        @property
        def std(self):
            return np.sqrt(self.var)

        @property
        def shape(self):
            return self._M.shape


    class ZFilter:
        """
        y = (x-mean)/std
        using running estimates of mean,std
        """

        def __init__(self, shape, demean=True, destd=True, clip=10.0):
            self.demean = demean
            self.destd = destd
            self.clip = clip

            self.rs = RunningStat(shape)

        def __call__(self, x, update=True):
            if update: self.rs.push(x)
            if self.demean:
                x = x - self.rs.mean
            if self.destd:
                x = x / (self.rs.std + 1e-8)
            if self.clip:
                x = np.clip(x, -self.clip, self.clip)
            return x

        def output_shape(self, input_space):
            return input_space.shape


    class ActorCritic(nn.Module):
        def __init__(self, num_inputs, num_outputs, layer_norm=True):
            super(ActorCritic, self).__init__()

            self.actor_fc1 = nn.Linear(num_inputs, 128)
            self.actor_fc2 = nn.Linear(128, config.hid_num)
            self.actor_fc3 = nn.Linear(config.hid_num, num_outputs)
            self.actor_logstd = nn.Parameter(torch.zeros(1, num_outputs))
            self.critic_fc1 = nn.Linear(num_inputs, 128)
            self.critic_fc2 = nn.Linear(128, 128)
            self.critic_fc3 = nn.Linear(128, 1)

            if layer_norm:
                self.layer_norm(self.actor_fc1, std=1.0)
                self.layer_norm(self.actor_fc2, std=1.0)
                self.layer_norm(self.actor_fc3, std=0.01)

                self.layer_norm(self.critic_fc1, std=1.0)
                self.layer_norm(self.critic_fc2, std=1.0)
                self.layer_norm(self.critic_fc3, std=1.0)

        @staticmethod
        def layer_norm(layer, std=1.0, bias_const=0.0):
            torch.nn.init.orthogonal_(layer.weight, std)
            torch.nn.init.constant_(layer.bias, bias_const)

        def forward(self, states):
            """
            run policy network (actor) as well as value network (critic)
            :param states: a Tensor2 represents states
            :return: 3 Tensor2
            """
            action_mean, action_logstd = self._forward_actor(states)
            critic_value = self._forward_critic(states)
            return action_mean, action_logstd, critic_value

        def _forward_actor(self, states):
            x = torch.tanh(self.actor_fc1(states))
            x = torch.tanh(self.actor_fc2(x))
            x = F.dropout(x, p=config.drop_prob, training=self.training)
            action_mean = torch.tanh(self.actor_fc3(x))
            action_logstd = self.actor_logstd.expand_as(action_mean)
            return action_mean, action_logstd

        def _forward_critic(self, states):
            x = torch.tanh(self.critic_fc1(states))
            x = torch.tanh(self.critic_fc2(x))
            critic_value = self.critic_fc3(x)
            return critic_value

        def select_action(self, action_mean, action_logstd, return_logproba=True):
            """
            given mean and std, sample an action from normal(mean, std)
            also returns probability of the given chosen
            """
            action_std = torch.exp(action_logstd)
            action = torch.normal(action_mean, action_std)
            if return_logproba:
                logproba = self._normal_logproba(action, action_mean, action_logstd, action_std)
            return action, logproba

        @staticmethod
        def _normal_logproba(x, mean, logstd, std=None):
            if std is None:
                std = torch.exp(logstd)

            std_sq = std.pow(2)
            logproba = - 0.5 * math.log(2 * math.pi) - logstd - (x - mean).pow(2) / (2 * std_sq)
            return logproba.sum(1)

        def get_logproba(self, states, actions):
            """
            return probability of chosen the given actions under corresponding states of current network
            :param states: Tensor
            :param actions: Tensor
            """
            action_mean, action_logstd = self._forward_actor(states)
            action_mean = action_mean.cpu()
            action_logstd = action_logstd.cpu()
            #print(actions,action_mean,action_logstd.cpu())
            logproba = self._normal_logproba(actions.cpu(), action_mean, action_logstd.cpu())
            return logproba


    class Memory(object):
        def __init__(self):
            self.memory = []

        def push(self, *args):
            self.memory.append(Transition(*args))

        def sample(self):
            return Transition(*zip(*self.memory))

        def __len__(self):
            return len(self.memory)

    env = gym.make(ENV_NAME)  
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    
    if USE_GPU:
        network = ActorCritic(num_inputs, num_actions, layer_norm=args.layer_norm).cuda()
        print('using GPU-{}'.format(config.use_gpu))
    else:
        network = ActorCritic(num_inputs, num_actions, layer_norm=args.layer_norm)
    network.train()
    def ppo(args):
        env = gym.make(args.env_name)
        num_inputs = env.observation_space.shape[0]
        num_actions = env.action_space.shape[0]

        env.seed(args.seed)
        torch.manual_seed(args.seed)

        #network = ActorCritic(num_inputs, num_actions, layer_norm=args.layer_norm)
        optimizer = opt.Adam(network.parameters(), lr=args.lr)

        running_state = ZFilter((num_inputs,), clip=5.0)

        # record average 1-round cumulative reward in every episode
        reward_record = []
        global_steps = 0

        lr_now = args.lr
        clip_now = args.clip

        for i_episode in range(args.num_episode):
            # step1: perform current policy to collect trajectories
            # this is an on-policy method!
            memory = Memory()
            num_steps = 0
            reward_list = []
            len_list = []
            while num_steps < args.batch_size:
                state = env.reset()
                if args.state_norm:
                    state = running_state(state)
                reward_sum = 0
                for t in range(args.max_step_per_round):
                    if USE_GPU:
                        action_mean, action_logstd, value = network(Tensor(state).float().unsqueeze(0).cuda())
                    else:
                        action_mean, action_logstd, value = network(Tensor(state).float().unsqueeze(0))
                    '''action dropout'''
                    action, logproba = network.select_action(action_mean, action_logstd)
                    action = action.cpu().data.numpy()[0]
                    logproba = logproba.cpu().data.numpy()[0]

                    next_state, reward, done, _ = env.step(action)

                    reward_sum += reward
                    if args.state_norm:
                        next_state = running_state(next_state)
                    mask = 0 if done else 1

                    memory.push(state, value, action, logproba, mask, next_state, reward)

                    if done:
                        break

                    state = next_state
                        
                num_steps += (t + 1)
                global_steps += (t + 1)
                reward_list.append(reward_sum)
                len_list.append(t + 1)
            reward_record.append({
                'episode': i_episode, 
                'steps': global_steps, 
                'meanepreward': np.mean(reward_list), 
                'meaneplen': np.mean(len_list)})
            rwds.extend(reward_list)
            batch = memory.sample()
            batch_size = len(memory)

            # step2: extract variables from trajectories
            rewards = Tensor(batch.reward).float()
            values = Tensor(batch.value).float()
            masks = Tensor(batch.mask).float()
            actions = Tensor(batch.action).float()
            states = Tensor(batch.state).float()
            oldlogproba = Tensor(batch.logproba).float()

            returns = Tensor(batch_size).float()
            deltas = Tensor(batch_size).float()
            advantages = Tensor(batch_size).float()

            prev_return = 0.
            prev_value = 0.
            prev_advantage = 0.
            for i in reversed(range(batch_size)):
                returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
                deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values[i]
                # ref: https://arxiv.org/pdf/1506.02438.pdf (generalization advantage estimate)
                advantages[i] = deltas[i] + args.gamma * args.lamda * prev_advantage * masks[i]

                prev_return = returns[i]
                prev_value = values[i]
                prev_advantage = advantages[i]
            if args.advantage_norm:
                advantages = (advantages - advantages.mean()) / (advantages.std() + EPS)

            for i_epoch in range(int(args.num_epoch * batch_size / args.minibatch_size)):
                # sample from current batch
                minibatch_ind = np.random.choice(batch_size, args.minibatch_size, replace=False)
                minibatch_states = states[minibatch_ind]
                minibatch_actions = actions[minibatch_ind]
                minibatch_oldlogproba = oldlogproba[minibatch_ind]
                minibatch_newlogproba = network.get_logproba(minibatch_states.cuda(), minibatch_actions.cuda()).cpu()
                minibatch_advantages = advantages[minibatch_ind]
                minibatch_returns = returns[minibatch_ind]
                minibatch_newvalues = network._forward_critic(minibatch_states.cuda()).cpu().flatten()

                ratio =  torch.exp(minibatch_newlogproba - minibatch_oldlogproba)
                surr1 = ratio * minibatch_advantages
                surr2 = ratio.clamp(1 - clip_now, 1 + clip_now) * minibatch_advantages
                loss_surr_cpu = - torch.mean(torch.min(surr1, surr2))

                if args.lossvalue_norm:
                    minibatch_return_6std = 6 * minibatch_returns.std()
                    loss_value_cpu = torch.mean((minibatch_newvalues - minibatch_returns).pow(2)) / minibatch_return_6std
                else:
                    loss_value_cpu = torch.mean((minibatch_newvalues - minibatch_returns).pow(2))

                loss_entropy_cpu = torch.mean(torch.exp(minibatch_newlogproba) * minibatch_newlogproba)

                total_loss_cpu = loss_surr_cpu + args.loss_coeff_value * loss_value_cpu + args.loss_coeff_entropy * loss_entropy_cpu
                optimizer.zero_grad()
                total_loss_cpu.backward()
                optimizer.step()

                
                

            if args.schedule_clip == 'linear':
                ep_ratio = 1 - (i_episode / args.num_episode)
                clip_now = args.clip * ep_ratio

            if args.schedule_adam == 'linear':
                ep_ratio = 1 - (i_episode / args.num_episode)
                lr_now = args.lr * ep_ratio
                # set learning rate
                # ref: https://stackoverflow.com/questions/48324152/
                for g in optimizer.param_groups:
                    g['lr'] = lr_now

            if i_episode % args.log_num_episode == 0:
                print("total loss cc",i_episode,reward_record[-1]['meanepreward'],total_loss_cpu.data,loss_surr_cpu.data,loss_value_cpu.data,loss_entropy_cpu.data)
                print('-----------------')

        return reward_record

    def test(args):
        record_dfs = []
        for i in range(args.num_parallel_run):
            args.seed += 1
            reward_record = pd.DataFrame(ppo(args))
            reward_record['#parallel_run'] = i
            record_dfs.append(reward_record)
        record_dfs = pd.concat(record_dfs, axis=0)
        record_dfs.to_csv(joindir(RESULT_DIR, 'ppo-record-{}.csv'.format(args.env_name)))

    if __name__ == '__main__':
        for envname in [ENV_NAME]:
            args.env_name = envname
            test(args)

    torch.save(network.state_dict(),ENV_NAME.split('-')[0] + '/CheckPoints/checkpoint_large_{0}hidden_{1}drop_prob_{2}repeat'.format(config.hid_num,config.drop_prob,repeat)) 
    np.savetxt(ENV_NAME.split('-')[0] + '/Rwds/rwds_large_{0}hidden_{1}drop_prob_{2}repeat'.format(config.hid_num,config.drop_prob,repeat),rwds)
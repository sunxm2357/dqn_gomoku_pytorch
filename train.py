import torch
import numpy as np
from dqn import *
import gym
import random
import gym_gomoku
from collections import namedtuple
from itertools import count
from tensorboardX import SummaryWriter
from options import *
from test import *
import pdb

Transition = namedtuple('Transition', ('state', "action", "next_state", "reward", "done"))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(Transition(*args))
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class EpsSchedule(object):
    def __init__(self, eps_start, eps_end, num_episodes):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.step = 2./num_episodes

    def get(self, episodes):
        eps = max([self.eps_end, self.eps_start - self.step * episodes])
        return eps


def select_action(env, epochs_done, EPS, model):
    state = np.array(env.state.board.board_state)
    r = random.random()
    eps = EPS.get(epochs_done)
    valid_action = env.action_space.valid_spaces
    if r < eps:
        # choose random
        return env.action_space.sample() # int
    else:
        best_action = model.choose_action(state, valid_action) # int
        if best_action != -1:
            return best_action
        else:
            raise ValueError('No valid action')


def visualize(episode_duration, episode_returns, writer):
    # calculate the average
    # durations_t = torch.FloatTensor(episode_duration)
    if len(episode_duration) >= 100:
        last = episode_duration[-100:]
        mean = sum(last) / float(len(last))
        last_returns = episode_returns[-100:]
        mean_return = sum(last_returns) / float(len(last_returns))
    else:
        mean = sum(episode_duration) / float(len(episode_duration))
        mean_return = sum(episode_returns) / float(len(episode_returns))

    # print on the screen
    print('Episode: %d, Duration: %d, Avg_Duration: %03f, Reward: %d, Avg_Reward: %03f' %
          (len(episode_duration), episode_duration[-1], mean, episode_returns[-1], mean_return))

    # add to writer
    writer.add_scalar('Current_Duration', episode_duration[-1], len(episode_duration))
    writer.add_scalar('Avg_Duration_over_last_100_games', mean, len(episode_duration))
    writer.add_scalar('Current_Reward', episode_returns[-1], len(episode_returns))
    writer.add_scalar('Avg_Reward_over_last_100_games', mean_return, len(episode_returns))


def main():
    opt = TrainOptions().parse()
    model = DQNModel()
    model.initialize(opt)
    EPS = EpsSchedule(opt.eps_start, opt.eps_end, opt.num_episodes) # epsilon scheduler
    memory = ReplayMemory(opt.capacity)
    writer = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.name))

    env = gym.make('Gomoku%dx%d-v0'%(opt.board_size, opt.board_size))
    env.reset()
    num_updates = 0
    episode_duration = []
    episode_returns = []
    for i_episode in range(model.start_episode, opt.num_episodes):
        env.reset()
        current_state = np.array(env.state.board.board_state)
        returns = 0
        t = 0
        while True:
            # print(t)
            # generate the samples
            action = select_action(env, i_episode, EPS, model)
            next_state, reward, done, _ = env.step(action)
            returns += reward * 10
            memory.push(current_state, action, next_state, reward, done)
            current_state = next_state
            # pdb.set_trace()
            # train the model
            t = t + 1
            if len(memory) < opt.begin_length:
                if done:
                    episode_duration.append(t)
                    episode_returns.append(returns)
                    visualize(episode_duration, episode_returns, writer)
                    if i_episode % opt.save_freq == 0:
                        model.save(i_episode, i_episode)
                        model.save('latest', i_episode)
                else:
                    continue
            else:
                print(len(memory))
                transitions = memory.sample(opt.batch_size)
                batch = Transition(*zip(*transitions))
                model.set_input(batch)
                model.optimize()
                num_updates += 1
                if num_updates % opt.target_update_freq == 0:
                    model.update_target()
            if done:
                episode_duration.append(t)
                episode_returns.append(returns)
                visualize(episode_duration, episode_returns, writer)
                if i_episode % opt.save_freq == 0:
                    model.save(i_episode, i_episode)
                    model.save('latest', i_episode)
                    for name, param in model.learn_Q.named_parameters():
                        writer.add_histogram('learn_'+name, param.clone().cpu().data.numpy(), i_episode)
                    for name, param in model.target_Q.named_parameters():
                        writer.add_histogram('target_'+name, param.clone().cpu().data.numpy(), i_episode)
                if i_episode % opt.val_freq == 0:
                    play_length, val_reward = validation(model, opt, i_episode)
                    writer.add_scalar('val_play_length', play_length, len(episode_duration))
                    writer.add_scalar('val_reward', val_reward, len(episode_duration))
                break


if __name__ == "__main__":
    main()
import torch
import numpy as np
from dqn import *
import gym
import random
import gym_gomoku
from collections import namedtuple
from itertools import count

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
    if r < eps:
        # choose random
        return env.action_space.sample() # int
    else:
        best_action = model.choose_action(state) # int
        if not best_action in env.action_space.valid_spaces:
            best_action = env.action_space.sample()
        return best_action

def main():
    # TODO, def opt
    opt = None
    model = DQNModel()
    model.initialize(opt)
    EPS = EpsSchedule(opt.eps_start, opt.eps_end, opt.num_episodes)
    memory = ReplayMemory(opt.capacity)

    env = gym.make('Gomoku%dx%d-v0'%(opt.board_size, opt.board_size))
    env.reset()
    num_updates = 0
    for i_episode in range(model.start_episode, opt.num_episodes):
        env.reset()
        current_state = np.array(env.state.board.board_state)
        for t in count():
            # generate the samples
            action = select_action(env, i_episode, EPS, model)
            next_state, reward, done, _ = env.step(action)
            memory.push(current_state, action, next_state, reward, done)
            current_state = next_state

            # train the model
            if len(memory) < opt.batch_size:
                continue
            else:
                transitions = memory.sample(opt.batch_size)
                batch = Transition(*zip(*transitions))
                model.set_input(batch)
                model.optimize()
                num_updates += 1
                if num_updates % opt.target_update_freq == 0:
                    model.update_target()

                # TODO: add visual


if __name__ == "__main__":
    main()
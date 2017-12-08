import torch
import numpy as np
from dqn import *
import gym
import gym_gomoku
import pdb


def select_action_val(env, model):
    state = np.array(env.state.board.board_state)
    valid_action = env.action_space.valid_spaces
    best_action = model.choose_action(state, valid_action) # int
    if best_action != -1:
        return best_action
    else:
        raise ValueError('No valid action')


def validation(model, opt, episode_idx):
    env = gym.make('Gomoku%dx%d-v0' % (opt.board_size, opt.board_size))
    env.reset()
    with open(opt.val_file, 'a') as f:
        f.write('====================after episode %d=========================\n' % episode_idx)
    t = 0
    while True:
        # print(t)
        # generate the samples
        with open(opt.val_file, 'a') as f:
            f.write(repr(env.state))
        action = select_action_val(env, model)
        next_state, reward, done, _ = env.step(action)
        t = t + 1
        env.render()
        if done:
            with open(opt.val_file, 'a') as f:
                f.write(repr(env.state))
            return t, reward

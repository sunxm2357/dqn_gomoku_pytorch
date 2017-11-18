import torch
import numpy as np
# import tensorboardX
from torch.autograd import Variable
from networks import *
from collections import namedtuple
import os

class DQNModel(object):
    def name(self):
        return "DQN_Model"

    def initialize(self, opt):
        self.gpu_ids = opt.gpu_ids
        self.board_width = opt.board_width
        self.units = opt.units
        self.is_train = opt.is_train()
        self.save_dir = opt.save_dir
        self.batch_size = opt.batch_size
        self.learn_Q = define_dqn(self.board_width, self.units)
        if self.is_train:
            self.start_episode = 1
            self.target_Q = define_dqn(self.board_width, self.units)
            self.update_target()
            self.MSELoss = torch.nn.MSELoss()
            self.optimizer = torch.optim.Adam(self.learn_Q.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        if not self.is_train or opt.continue_train:
            self.load(opt.which_epoch)

    def forward_learn(self, state):
        q_s = self.learn_Q.forward(state)
        return q_s

    def forward_target(self, state):d
        q_s = self.target_Q.forward(state)
        return q_s

    def set_input(self, batch):
        use_gpu = len(self.gpu_ids) > 0
        state_tuple = batch.state
        action_tuple = batch.action
        next_state_tuple = batch.next_state
        reward_tuple = batch.reward
        done_list = list(batch.done)

        # set up state tensor
        state = torch.from_numpy(np.expand_dims(np.stack(state_tuple, axis=0), axis=1))
        if use_gpu:
            state = state.cuda()
        self.state = Variable(state)

        # set up action tensor
        action = torch.from_numpy(np.stack(action_tuple, axis=0))
        if use_gpu:
            action = action.cuda()
        self.action = Variable(action)

        # set up reward tensor
        reward = torch.from_numpy(np.stack(reward_tuple, axis=0))
        if use_gpu:
            reward = reward.cuda()
        self.reward = Variable(reward)

        # set up next_state tensor
        self.non_final_idx = []
        non_final_next_state = []
        for i, done in enumerate(done_list):
            if not done:
                self.non_final_idx.append(i)
                non_final_next_state.append(next_state_tuple(i))
        next_state = torch.from_numpy(np.expand_dims(np.stack(non_final_next_state , axis=0), axis=1))
        if use_gpu:
            next_state = next_state.cuda()
        self.next_state = Variable(next_state, volatile=True)


    def backward(self):
        q_sa = self.forward_learn(self.state).gather(1, self.action)
        q_nsa = self.forward_target(self.next_state).max(1)
        if len(self.gpu_ids) > 0:
            next_state_values = Variable(torch.zeros(self.batch_size).float().cuda())
        else:
            next_state_values = Variable(torch.zeros(self.batch_size).float())
        for i, idx in enumerate(self.non_final_idx):
            next_state_values[idx] = q_nsa[i]

        expected_state_action_values = next_state_values + self.reward
        self.loss = self.MSELoss(q_sa, expected_state_action_values)
        self.loss.backward()

    def optimize(self):
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def choose_action(self, state):
        state = np.expand_dims(np.expand_dims(state, axis=0), axis=0)
        state = Variable(state, volatile=True)
        if len(self.gpu_ids)>0:
            q_s = self.forward_learn(state).data.cpu().numpy()
        else:
            q_s = self.forward_learn(state).data.numpy()
        best_action = np.argmax(q_s, axis=1)[0]
        return best_action

    def update_target(self):
        self.target_Q.load_state_dict(self.learn_Q.state_dict())

    def save(self, label, episode):
        current_state = {
            "episode": episode,
            "learn_Q": self.learn_Q.cpu().state_dict(),
            "target_Q": self.target_Q.cpu().state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        save_filename = '%s_model.pth.tar' % (label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(current_state, save_path)
        if len(self.gpu_ids) and torch.cuda.is_available():
            self.learn_Q.cuda(device_id=self.gpu_ids[0])
            self.target_Q.cuda(device_id=self.gpu_ids[0])

    def load(self, epoch_label):
        save_filename = '%s_model.pth.tar' % (epoch_label)
        save_path = os.path.join(self.save_dir, save_filename)
        if os.path.isfile(save_path):
            print("=> loading snapshot from {}".format(save_path))
            snapshot = torch.load(save_path)
            self.start_episode = snapshot["episode"] + 1
            self.learn_Q.load_state_dict(snapshot['learn_Q'])
            if self.is_train:
                self.target_Q.load_state_dict(snapshot['target_Q'])
                self.optimizer.load_state_dict(snapshot["optimizer"])




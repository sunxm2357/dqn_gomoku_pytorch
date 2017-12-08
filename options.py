# TODO
# is_train/
# gpu_ids/
# num_episodes/
# board_size/
# batch_size/
# units/
# lr/
# beta1/
# eps_start/
# eps_end/
# num_episodes/
# tensorboard_dir/
# name/
# target_update_freq/

import argparse
import os
from utils import *
import torch
import copy

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--board_size', type=int, default=19, help='the size of the board [19]')
        self.parser.add_argument('--units', type=int, default=512, help="the units for nn, 512(default) for fc, 16 for conv")
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--tensorboard_dir', type=str, default='./tb', help='models are saved here')
        self.parser.add_argument('--type', type=str, default='fc', help = 'what kind of layers [conv|fc]')
        # self.parser.add_argument('--model', type=str, default='dqn', help='the model to run')

        # TODO: add or delete
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.is_train = self.is_train   # train or test


        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        makedir(expr_dir)
        tb_dir = os.path.join(self.opt.tensorboard_dir, self.opt.name)
        makedir(tb_dir)
        self.opt.save_dir = expr_dir

        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')

        self.opt.val_file = os.path.join(expr_dir, 'val.txt')

        return self.opt


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument("--batch_size", type=int, dest="batch_size", default=40, help="Mini-batch size")
        self.parser.add_argument("--lr", type=float, dest="lr", default=0.00025, help="Base Learning Rate")
        self.parser.add_argument("--num_episodes", type=int, default=1000000, help="Number of iterations")
        self.parser.add_argument('--target_update_freq', type=int, default=500, help='frequency of updating the target network')
        self.parser.add_argument('--save_freq', type=int, default=500, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--val_freq', type=int, default=2000, help='frequency of validation')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--eps_start', type=float, default=0.9, help='the beginning of epsilon, 0.9 by default')
        self.parser.add_argument('--eps_end', type=float, default=0.05, help='the end of epsilon, 0.05 by default')
        self.parser.add_argument('--capacity', type=int, default=100000, help='the capacity of replay buffer')
        self.parser.add_argument('--begin_length', type=int, default=1000, help='begin training when the buffer increases to this length')

        self.is_train = True


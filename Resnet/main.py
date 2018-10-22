import os
import argparse
import gym
import numpy as np
import torch
import torch.cuda
import torch.multiprocessing as _mp
import torch.nn as nn
from setup_env import setup_env
#from model import ActorCritic
from shared_adam import SharedAdam
from Train import train, test
import torchvision.models as models
from new_model2 import Resnet
SAVEPATH = os.getcwd() + '/save/mario_a3c_params.pkl'

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.9,
                    help='discount factor for rewards (default: 0.9)')
parser.add_argument('--tau', type=float, default=1.00,
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=250,
                    help='value loss coefficient (default: 50)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 4)')
parser.add_argument('--num-processes', type=int, default=8,
                    help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=50,
                    help='number of forward steps in A3C (default: 50)')
parser.add_argument('--max-episode-length', type=int, default=10000,
                    help='maximum length of an episode (default: 1000000)')
parser.add_argument('--env-name', default='SuperMarioBros-1-1-v0',
                    help='environment to train on (default: SuperMarioBros-1-4-v0)')
parser.add_argument('--no-shared', default=False,
                    help='use an optimizer without shared momentum.')
parser.add_argument('--use-cuda',default=False,
                    help='run on gpu.')
parser.add_argument('--save-interval', type=int, default=100,
                    help='model save interval (default: 10)')
parser.add_argument('--save-path',default=SAVEPATH,
                    help='model save interval (default: {})'.format(SAVEPATH))
parser.add_argument('--non-sample', type=int,default=2,
help='number of non sampling processes (default: 2)')

mp = _mp.get_context('spawn')

print("Cuda: " + str(torch.cuda.is_available()))

if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'

    args = parser.parse_args()
    env = setup_env(args.env_name)

    shared_model= Resnet(1, env.action_space.n)

    if args.use_cuda:
        shared_model.cuda()

    shared_model.share_memory()

    if os.path.isfile(args.save_path):
        #print('Loading A3C parametets ...')
        shared_model.load_state_dict(torch.load(args.save_path))

    optimizer = SharedAdam(filter(lambda p: p.requires_grad, shared_model.parameters()), lr=args.lr)
    optimizer.share_memory()

    print ("No of available cores : {}".format(mp.cpu_count())) 

    processes = []

    counter = mp.Value('i', 0)
    lock = mp.Lock()

    p = mp.Process(target=test, args=(args.num_processes, args, shared_model, counter))
    
    p.start()
    processes.append(p)

    num_procs = args.num_processes
    no_sample = args.non_sample
   
    if args.num_processes > 1:
        num_procs = args.num_processes - 1    

    sample_val = num_procs - no_sample

    for rank in range(0, num_procs):
        if rank < sample_val:                           # select random action
            p = mp.Process(target=train, args=(rank, args, shared_model, counter, lock, optimizer))
        else:                                           # select best action
            p = mp.Process(target=train, args=(rank, args, shared_model, counter, lock, optimizer, False))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
import argparse
import os
import random
import subprocess
import numpy as np
import torch

parser = argparse.ArgumentParser(description='Run the multi-agent sacd with various parameters.')
# State arguments.
if torch.cuda.is_available():
    parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--test-amb-init', action='store_true', default=False)
parser.add_argument('--test-final', action='store_true', default=False)
parser.add_argument('--train-script', default='train_multi_sacd.py')
parser.add_argument('--test-script', default='evaluate_multi_sacd.py')
parser.add_argument('--task', default='simple_tag')
parser.add_argument('--num-good-agents', type=int, default=0)
parser.add_argument('--num-adversaries', type=int, default=0)
parser.add_argument('--epoch', type=int, default=30)
parser.add_argument('--date', type=str, default='0000')
# actor and critic learning rates
parser.add_argument('--alr', type=float, default=1e-3)
parser.add_argument('--clr', type=float, default=1e-3)
parser.add_argument('--layer-num', type=int, default=2)
parser.add_argument('--obs-radius', type=float, default=float('inf'))
parser.add_argument('--with-comm', action='store_true', default=False)
parser.add_argument('--centralized', action='store_true', default=False)
parser.add_argument('--others-model', action='store_true', default=False)
parser.add_argument('--rew-shape', action='store_true', default=False)
parser.add_argument('--grads-logging', action='store_true', default=False)
parser.add_argument('--only-self-pred', action='store_true', default=False)
parser.add_argument('--curiosity', action='store_true', default=False)
parser.add_argument('--ma-curiosity', action='store_true', default=False)
parser.add_argument('--self-adv', action='store_true', default=False)
parser.add_argument('--seeds', type=int, nargs='+', help='a list of random seeds', required=True)
parser.add_argument('--wm-noise-level', type=float, default=0.0)
args = parser.parse_args()

task_dir = args.task + "/" if "simple_adversary" not in args.task else "simple_adv/"
base_dir = '/scr/zixianma/multiagent/' + task_dir if torch.cuda.is_available() else 'log/' + task_dir
if not os.path.exists(base_dir):
    os.makedirs(base_dir)
num_good_agents = args.num_good_agents
num_adversaries = args.num_adversaries

for rn_seed in args.seeds:
    for intr_rew in [1]:
        if intr_rew == 0 or args.rew_shape:   
            intr_rew_configs = ['000']
        else:
            # collaborative settings, no intr rew from adversaries
            if 'spread' in args.task or args.obs_radius == float('inf') or args.others_model:
                intr_rew_configs = ['101'] 
            else:
                intr_rew_configs = ['101']
        # overwrite intr_rew_configs under special conditions
        if args.only_self_pred:
            intr_rew_configs = ['self']
        elif args.curiosity:
            intr_rew_configs = ['curiosity']
        elif args.ma_curiosity:
            intr_rew_configs = ['ma_curiosity']
        elif args.self_adv:
            intr_rew_configs = ['self_adv']
        for config in intr_rew_configs:
            log_dir =  base_dir + args.date + '_' + args.task + '_' + str(num_good_agents) + 'v' + str(num_adversaries) \
                        + '_eps_' + str(args.epoch) + '_obs_radius_' \
                        + (str(args.obs_radius) if args.obs_radius == float('inf') else str(int(args.obs_radius))) 
            if "comm" in args.task:
                log_dir += '_comm' if args.with_comm else '_no_comm'
            log_dir += '_intr_rew_' + config + '_rep_' + str(rn_seed)
            train_params = {
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'task': args.task + '_in',
                'epoch': args.epoch,
                'buffer-size': 2000000,
                'step-per-epoch': 100,
                'rew-norm': 1,
                'actor-lr': args.alr,
                'critic-lr': args.clr,
                'layer-num': args.layer_num,
                'num-good-agents': num_good_agents,
                'num-adversaries': num_adversaries,
                'obs-radius': args.obs_radius,
                'intr-rew': config,
                'wm-noise-level': args.wm_noise_level,
                'logdir': log_dir,
                'seed': rn_seed
            }
            arguments = ' '.join(['--' + k + ' ' + str(train_params[k]) for k in train_params])
            train = 'python ' + args.train_script 
            train += ' --save-models --benchmark' 
            # if rn_seed == 0:
            # train += ' --wandb-enabled'
            if args.with_comm:
                train += ' --with-comm'
            if args.centralized:
                train += ' --centralized'
            if args.others_model:
                train += ' --others-model'
            if args.grads_logging:
                train += ' --grads-logging'
            if args.rew_shape:
                train += ' --rew-shape'
            if args.only_self_pred:
                train += ' --only-self-pred'
            train += ' ' + arguments
            print('\n' + '*' * 45  + 'Training' + '*' * 45  + '\n')
            print(train)
            subprocess.call(train, shell=True)

            if args.test:
                test_params = {
                'logdir': log_dir,
                'savedir': 'result',
                # 'test-final': True if args.test_final else False
                } 
                arguments = ' '.join(['--' + k + ' ' + str(test_params[k]) for k in test_params])
                test = 'python ' + args.test_script
                test += ' ' + arguments
                print('\n' + '*' * 45  + 'Testing' + '*' * 45  + '\n')
                print(test)
                subprocess.call(test, shell=True)
            
            if args.test_amb_init:
                test_params = {
                'logdir': log_dir,
                'savedir': 'result',
                'amb-init': 1,
                # 'test-final': True if args.test_final else False
                } 
                arguments = ' '.join(['--' + k + ' ' + str(test_params[k]) for k in test_params])
                test = 'python ' + args.test_script
                test += ' ' + arguments
                print('\n' + '*' * 45  + 'Testing' + '*' * 45  + '\n')
                print(test)
                subprocess.call(test, shell=True)


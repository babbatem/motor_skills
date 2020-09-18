import itertools
import argparse
import datetime
import os
import sys
import re
import time
import numpy as np
import argparse

"""
TODOs
[x] script body
[x] env_kwargs added to config file
[] remove demonstration bits
[] sbatch > qsub
[] request a gpu?
[] loop over start states (for this particular experiment - one seed each?)

"""

def filldict(listKeys, listValues):
	mydict = {}
	for key, value in zip(listKeys, listValues):
		 mydict[key] = value
	return mydict

def generate_script_body(param_dict):
	script_body=\
'''#!/bin/bash

#SBATCH --time=4:00:00

#SBATCH -N 1
#SBATCH -c 8
#SBATCH -J cip-learn
#SBATCH --mem=8G

#SBATCH -o cip-learn-%j.out
#SBATCH -e cip-learn-%j.out

cd /users/babbatem/
source .bashrc
source load_mods.sh

cd motor_skills
python3 my_job_script.py --config {} --output {}

'''
	script_body=script_body.format(param_dict['config'],
								   param_dict['output'])
	return script_body

def get_config_file_npg():
	config= \
"""{
# general inputs

'env'           :   '%s',
'env_kwargs'    :   '%s',
'algorithm'     :   'NPG',
'seed'          :   %i,
'num_cpu'       :   3,
'save_freq'     :   25,
'eval_rollouts' :   1,

# RL parameters (all params related to PG, value function, DAPG etc.)
'policy_size'   :   (32, 32),
'vf_batch_size' :   64,
'vf_epochs'     :   2,
'vf_learn_rate' :   1e-3,
'rl_step_size'  :   0.05,
'rl_gamma'      :   0.995,
'rl_gae'        :   0.97,
'rl_num_traj'   :   20,
'rl_num_iter'   :   10,
'lam_0'         :   0,
'lam_1'         :   0,
'init_log_std'  :   0,
}
"""
	return config

def submit(param_dict):
	script_body = generate_script_body(param_dict)

	objectname = param_dict['algo'] + '-' \
	 		   + param_dict['env-short'] + '-' \
			   + str(param_dict['seed'])

	jobfile = "scripts/{}/{}".format(param_dict['name'], objectname)
	with open(jobfile, 'w') as f:
		f.write(script_body)
	cmd="sbatch {}".format(jobfile)
	os.system(cmd)
	return 0

def main(args):

	KEYS = ['seed', 'env', 'algo', 'config', 'output', 'name', 'env-short']
	SEEDS = np.arange(13)

	full_env_names_dict = {'cip': 'motor_skills:mj_jaco_door_cip-v0',
						   'naive': 'motor_skills:mj_jaco_door_naive-v0',
						   }
	full_env_name = full_env_names_dict[args.env]

	os.makedirs('exps' + '/' + args.exp_name, exist_ok=True)
	config_root = 'exps' + '/' + args.exp_name + '/' + args.env + '/configs/'
	output_root = 'exps' + '/' + args.exp_name + '/' + args.env + '/outputs/'
	os.makedirs('scripts/%s' % args.exp_name, exist_ok=True)
	os.makedirs(config_root, exist_ok=True)
	os.makedirs(output_root, exist_ok=True)

	k=0
	for i in range(len(SEEDS)):

		# get the config text
		if args.algo == 'dapg':
			config = get_config_file_dapg()
		elif args.algo == 'npg':
			config = get_config_file_npg()
		else:
			print('Invalid algorithm name [dapg, npg]')
			raise ValueError

		env_kwargs_string = "{'start_idx': %i}" % i

		config=config % (full_env_name, env_kwargs_string, SEEDS[i])
		config_path = config_root + args.algo + str(SEEDS[i]) + '.txt'
		config_writer = open(config_path,'w')
		config_writer.write(config)
		config_writer.close()

		output_path = output_root + args.algo + str(SEEDS[i])

		element = [SEEDS[i],
				   full_env_name,
				   args.algo,
				   config_path,
				   output_path,
				   args.exp_name,
				   args.env]

		param_dict = filldict(KEYS, element)
		submit(param_dict)
		k+=1
	print(k)

if __name__ == '__main__':
	parser=argparse.ArgumentParser()
	parser.add_argument('-t', '--test', action='store_true', help='don\'t submit, just count')
	parser.add_argument('-n', '--exp-name', required=True, type=str, help='parent directory for jobs')
	parser.add_argument('-g', '--gpu', action='store_true', help='request gpus')
	parser.add_argument('-e', '--env', type=str, help='microwave, drawer, or dynamic')
	parser.add_argument('-a', '--algo', type=str, help='dapg or npg')
	args=parser.parse_args()
	main(args)

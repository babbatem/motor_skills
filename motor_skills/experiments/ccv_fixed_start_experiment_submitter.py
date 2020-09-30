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
[x] sbatch > qsub
[] request a gpu?
[x] loop over start states (for this particular experiment - one seed each)
		if we get mismatch, run more.

[] seeds and learning rates
[] connecting to onager/tune/etc.
[] job name arg to batch script, or automatically name the output and err files.
"""

def filldict(listKeys, listValues):
	mydict = {}
	for key, value in zip(listKeys, listValues):
		 mydict[key] = value
	return mydict

def generate_script_body(param_dict):
	script_body=\
'''#!/bin/bash

#SBATCH --time=2:00:00

#SBATCH -N 1
#SBATCH -c 4
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
'env_kwargs'    :   %s,
'algorithm'     :   'NPG',
'seed'          :   %i,
'num_cpu'       :   4,
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
'rl_num_traj'   :   100,
'rl_num_iter'   :   100,
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
			   + str(param_dict['seed']) + '-' \
			   + str(param_dict['start'])

	jobfile = "scripts/{}/{}".format(param_dict['name'], objectname)
	with open(jobfile, 'w') as f:
		f.write(script_body)
	cmd="sbatch {}".format(jobfile)
	os.system(cmd)
	return 0

def main(args):

	KEYS = ['seed', 'start', 'env', 'algo', 'config', 'output', 'name', 'env-short']
	STARTS = np.arange(13)
	SEEDS = np.arange(3)

	full_env_names_dict = {'cip': 'motor_skills:mj_jaco_door_cip-v0',
						   'naive': 'motor_skills:mj_jaco_door_naive-v0',
						   }
	full_env_name = full_env_names_dict[args.env]

	try:
		os.makedirs('/users/babbatem/motor_skills/experiments/exps' + '/' + args.exp_name, exist_ok=True)
		config_root = '/users/babbatem/motor_skills/experiments/exps' + '/' + args.exp_name + '/' + args.env + '/configs/'
		output_root = '/users/babbatem/motor_skills/experiments/exps' + '/' + args.exp_name + '/' + args.env + '/outputs/'
		os.makedirs('scripts/%s' % args.exp_name, exist_ok=True)
		os.makedirs(config_root, exist_ok=True)
		os.makedirs(output_root, exist_ok=True)
	except Exception as e:
		print('failed to create experiment config and output dirs with error: ')
		print(e)
		print('you must be on your local machine. trying to create dirs locally. ')
		os.makedirs('/home/abba/msu_ws/src/motor_skills/experiments/exps' + '/' + args.exp_name, exist_ok=True)
		config_root = '/home/abba/msu_ws/src/motor_skills/experiments/exps' + '/' + args.exp_name + '/' + args.env + '/configs/'
		output_root = '/home/abba/msu_ws/src/motor_skills/experiments/exps' + '/' + args.exp_name + '/' + args.env + '/outputs/'
		os.makedirs('scripts/%s' % args.exp_name, exist_ok=True)
		os.makedirs(config_root, exist_ok=True)
		os.makedirs(output_root, exist_ok=True)


	k=0
	for i in range(len(STARTS)):
		for j in range(len(SEEDS)):

			# get the config text
			if args.algo == 'dapg':
				config = get_config_file_dapg()
			elif args.algo == 'npg':
				config = get_config_file_npg()
			else:
				print('Invalid algorithm name [dapg, npg]')
				raise ValueError

			env_kwargs_string = "{\"start_idx\": %i}" % i

			config=config % (full_env_name, env_kwargs_string, SEEDS[j])
			config_path = config_root + args.algo + str(SEEDS[j]) + '_' + str(STARTS[i]) + '.txt'
			config_writer = open(config_path,'w')
			config_writer.write(config)
			config_writer.close()

			output_path = output_root + args.algo + str(SEEDS[j]) + '_' + str(STARTS[i])

			element = [SEEDS[j],
					   STARTS[i],
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

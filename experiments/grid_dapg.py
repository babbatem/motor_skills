import itertools
import argparse
import datetime
import os
import sys
import re
import time
import numpy as np
import argparse

def filldict(listKeys, listValues):
	mydict = {}
	for key, value in zip(listKeys, listValues):
		 mydict[key] = value
	return mydict

def generate_script_body(param_dict):
	script_body='''#!/bin/bash
cd /home/babbatem/projects/skills_kin/ben_dapg
source /home/babbatem/envs/skills_kin/bin/activate
export GYM_ENV={}
echo $GYM_ENV
python my_job_script.py --config {} --output {}

'''
	script_body=script_body.format(param_dict['env'],
								   param_dict['config'],
								   param_dict['output'])
	return script_body

def get_config_file_dapg():
	config= \
"""{
# general inputs

'env'           :   '%s',
'algorithm'     :   'DAPG',
'seed'          :   %i,
'num_cpu'       :   3,
'save_freq'     :   25,
'eval_rollouts' :   1,

# Demonstration data and behavior cloning
'demo_file'     :   '%s',
'bc_batch_size' :   32,
'bc_epochs'     :   5,
'bc_learn_rate' :   1e-3,

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
'lam_0'         :   1e-2,
'lam_1'         :   0.95,
'init_log_std'  :   1
}
"""
	return config

def get_config_file_npg():
	config= \
"""{
# general inputs

'env'           :   '%s',
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
'init_log_std'  :   1,
}
"""
	return config

def submit(param_dict, job_details):
	script_body = generate_script_body(param_dict)

	objectname = param_dict['algo'] + '-' \
	 		   + param_dict['env-short'] + '-' \
			   + str(param_dict['seed'])

	jobfile = "scripts/{}/{}".format(param_dict['name'], objectname)
	with open(jobfile, 'w') as f:
		f.write(script_body)
	cmd="qsub {} {}".format(job_details, jobfile)
	os.system(cmd)
	return 0

def main(args):

	KEYS = ['seed', 'env', 'algo', 'config', 'output', 'name', 'env-short']
	SEEDS = np.arange(5)

	# TODO: make this mapping correct
	full_env_names_dict = {'drawer': 'kuka_gym:KukaDrawer-v0',
						   'microwave': 'kuka_gym:KukaCabinet-v0',
						   'dynamic': 'kuka_gym:KukaDynamic-v0'}
	full_env_name = full_env_names_dict[args.env]

	if args.gpu:
		request = '-l long -l vf=32G -l gpus=1 -q gpus*'
	else:
		request = '-l long -l vf=32G -pe smp 3'

	os.makedirs('experiments' + '/' + args.exp_name, exist_ok=True)
	config_root = 'experiments' + '/' + args.exp_name + '/' + args.env + '/configs/'
	output_root = 'experiments' + '/' + args.exp_name + '/' + args.env + '/outputs/'
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

		demo_path = '/home/babbatem/projects/skills_kin/sim/data/kuka_%s_demo.pickle'
		demo_path = demo_path % args.env

		if args.algo == 'dapg':
			config=config % (full_env_name, SEEDS[i], demo_path)
		else:
			config=config % (full_env_name, SEEDS[i])
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
		submit(param_dict, request)
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

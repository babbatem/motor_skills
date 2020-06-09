"""
This is a job script for running NPG/DAPG on hand tasks and other gym envs.
Note that DAPG generalizes PG and BC init + PG finetuning.
With appropriate settings of parameters, we can recover the full family.
"""

from mjrl.mjrl.utils.gym_env import GymEnv
from mjrl.mjrl.policies.gaussian_mlp import MLP
from mjrl.mjrl.baselines.quadratic_baseline import QuadraticBaseline
from mjrl.mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.mjrl.algos.npg_cg import NPG
from mjrl.mjrl.algos.dapg import DAPG
from mjrl.mjrl.algos.behavior_cloning import BC
from mjrl.mjrl.utils.train_agent import train_agent
from mjrl.mjrl.samplers.core import sample_paths
import os
import json
import mjrl.envs
# import mj_envs
import time as timer
import pickle
import argparse

# ===============================================================================
# Get command line arguments
# ===============================================================================

parser = argparse.ArgumentParser(description='Policy gradient algorithms with demonstration data.')
parser.add_argument('--output', type=str, required=True, help='location to store results')
parser.add_argument('--config', type=str, required=True, help='path to config file with exp params')
args = parser.parse_args()
JOB_DIR = args.output
if not os.path.exists(JOB_DIR):
    os.mkdir(JOB_DIR)
with open(args.config, 'r') as f:
    job_data = eval(f.read())
assert 'algorithm' in job_data.keys()
assert any([job_data['algorithm'] == a for a in ['NPG', 'BCRL', 'DAPG']])
job_data['lam_0'] = 0.0 if 'lam_0' not in job_data.keys() else job_data['lam_0']
job_data['lam_1'] = 0.0 if 'lam_1' not in job_data.keys() else job_data['lam_1']
EXP_FILE = JOB_DIR + '/job_config.json'
with open(EXP_FILE, 'w') as f:
    json.dump(job_data, f, indent=4)

# ===============================================================================
# Train Loop
# ===============================================================================

e = GymEnv(job_data['env'])
spec = e.spec

# pickle.dump(e.spec, open('envspec.pickle', 'wb'))
# import sys
# sys.exit()

# spec = pickle.load(open('envspec.pickle','rb'))
policy = MLP(spec,
             hidden_sizes=job_data['policy_size'],
             seed=job_data['seed'],
             init_log_std=job_data['init_log_std'])             
baseline = MLPBaseline(spec, reg_coef=1e-3, batch_size=job_data['vf_batch_size'],
                       epochs=job_data['vf_epochs'], learn_rate=job_data['vf_learn_rate'])

# Get demonstration data if necessary and behavior clone
if job_data['algorithm'] != 'NPG':
    print("========================================")
    print("Collecting expert demonstrations")
    print("========================================")
    demo_paths = pickle.load(open(job_data['demo_file'], 'rb'))

    bc_agent = BC(demo_paths, policy=policy, epochs=job_data['bc_epochs'], batch_size=job_data['bc_batch_size'],
                  lr=job_data['bc_learn_rate'], loss_type='MSE', set_transforms=False)
    in_shift, in_scale, out_shift, out_scale = bc_agent.compute_transformations()
    bc_agent.set_transformations(in_shift, in_scale, out_shift, out_scale)
    bc_agent.set_variance_with_data(out_scale)

    ts = timer.time()
    print("========================================")
    print("Running BC with expert demonstrations")
    print("========================================")
    bc_agent.train()
    print("========================================")
    print("BC training complete !!!")
    print("time taken = %f" % (timer.time() - ts))
    print("========================================")

    if job_data['eval_rollouts'] >= 1:
        # % with constant start state and goal, just 1 episode is sufficient.
        score = e.evaluate_policy(policy, num_episodes=1, mean_action=True, visual=False)
        print("Score with behavior cloning = %f" % score[0][0])

if job_data['algorithm'] != 'DAPG':
    # We throw away the demo data when training from scratch or fine-tuning with RL without explicit augmentation
    demo_paths = None

# % disconnect after creating top-level env.

# ===============================================================================
# RL Loop
# ===============================================================================
# e = GymEnv(job_data['env'])
# e=[]
rl_agent = DAPG(e, policy, baseline, demo_paths,
                normalized_step_size=job_data['rl_step_size'],
                lam_0=job_data['lam_0'], lam_1=job_data['lam_1'],
                seed=job_data['seed'], save_logs=True
                )

# import pybullet as p
# p.disconnect()
# del e

print("========================================")
print("Starting reinforcement learning phase")
print("========================================")

ts = timer.time()
train_agent(job_name=JOB_DIR,
            agent=rl_agent,
            seed=job_data['seed'],
            niter=job_data['rl_num_iter'],
            gamma=job_data['rl_gamma'],
            gae_lambda=job_data['rl_gae'],
            num_cpu=job_data['num_cpu'],
            sample_mode='trajectories',
            num_traj=job_data['rl_num_traj'],
            save_freq=job_data['save_freq'],
            evaluation_rollouts=job_data['eval_rollouts'])
print("time taken = %f" % (timer.time()-ts))

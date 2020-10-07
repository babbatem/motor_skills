import numpy as np
import gym
import pickle
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.quadratic_baseline import QuadraticBaseline
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.dapg import DAPG
from mjrl.utils import tensor_utils

def get_env(env, env_kwargs={}):

    # get the correct env behavior
    if type(env) == str:
        # env = GymEnv(env, vis=True)
        env = GymEnv(env, env_kwargs=env_kwargs)
    elif isinstance(env, GymEnv):
        env = env
    elif callable(env):
        env = env(**env_kwargs)
    else:
        print("Unsupported environment format")
        raise AttributeError

    return env

def do_replays(
		num_traj,
		env,
		policy,
		eval_mode = False,
		horizon = 1e6,
		base_seed = 123,
		env_kwargs={},
):
	"""
	:param num_traj:    number of trajectories (int)
	:param env:         environment (env class, str with env_name, or factory function)
	:param policy:      policy to use for action selection
	:param eval_mode:   use evaluation mode for action computation (bool)
	:param horizon:     max horizon length for rollout (<= env.horizon)
	:param base_seed:   base seed for rollouts (int)
	:param env_kwargs:  dictionary with parameters, will be passed to env generator
	:return:
	"""

	env_made=get_env(env, env_kwargs=env_kwargs)

	if base_seed is not None:
		env_made.set_seed(base_seed)
		np.random.seed(base_seed)
	else:
		np.random.seed()
	horizon = min(horizon, env_made.horizon)
	paths = []

	for ep in tqdm(range(num_traj)):

		env_kwargs['start_idx'] = ep

		del env_made
		env_made=get_env(env, env_kwargs=env_kwargs)

		# seeding
		if base_seed is not None:
			seed = base_seed + ep
			env_made.set_seed(seed)
			np.random.seed(seed)

		observations=[]
		actions=[]
		rewards=[]
		agent_infos = []
		env_infos = []

		o = env_made.reset()
		done = False
		t = 0

		while t < horizon and done != True:
			a, agent_info = policy.get_action(o)
			if eval_mode:
				a = agent_info['evaluation']

			# print(a[:6])
			env_info_base = env_made.get_env_infos()
			next_o, r, done, env_info_step = env_made.step(a)
			# below is important to ensure correct env_infos for the timestep
			env_info = env_info_step if env_info_base == {} else env_info_base
			observations.append(o)
			actions.append(a)
			rewards.append(r)
			agent_infos.append(agent_info)
			env_infos.append(env_info)
			o = next_o
			t += 1

		observations=np.array(observations)
		np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
		# print('Episode Total: ', sum(rewards))

		path = dict(
			observations=np.array(observations),
			actions=np.array(actions),
			rewards=np.array(rewards),
			agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
			env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
			terminated=done,
			start_idx=env_made.env.start_idx
		)
		paths.append(path)

	del(env_made)
	return paths

if __name__ == '__main__':
    # from motor_skills.envs.mj_jaco import MjJacoDoorImpedanceCIP
    # env=MjJacoDoorImpedanceCIP(vis=True)
    # f=open('experiments/cip/dev/policy_15.pickle','rb')
    f=open('experiments/cip/dev/policy_80.pickle','rb')
    # f=open('experiments/naive/dev/policy_95.pickle','rb')
    policy = pickle.load(f)
    num_traj = 22

    env='motor_skills:mj_jaco_door_cip-v0'
    # env='motor_skills:mj_jaco_door_naive-v0'

    try:
    	paths = np.load('REPLAY_PATHS_2020_09_17.npy', allow_pickle=True)
    except:
    	paths=do_replays(num_traj,
    			   env,
    			   policy,
    			   eval_mode = True,
    			   horizon = 1e6,
    			   base_seed = 1,
    			   env_kwargs= {'start_idx': 0, 'vis':True, 'n_steps':2000}
    			   )

    np.save('REPLAY_PATHS_2020_09_17_1000paths.npy', paths)

    env_made = get_env(env)
    succ_rate = env_made.env.evaluate_success(paths)
    print('success rate: ', succ_rate)

    n_total_grasps = len(env_made.env.cip.grasp_qs)
    successes_arr = np.zeros(n_total_grasps)
    times_sampled_arr = np.zeros(n_total_grasps)
    x = np.arange(n_total_grasps)
    for p in paths:
    	start_idx = p['start_idx']
    	times_sampled_arr[start_idx] += 1
    	if p['env_infos']['success'][-1]:
    		successes_arr[start_idx] += 1

    successes_arr = np.array(successes_arr)
    times_sampled_arr = np.array(times_sampled_arr)
    success_rate_arr = successes_arr / times_sampled_arr


    plt.subplot(121); plt.bar(x, times_sampled_arr); plt.title('times sampled')
    plt.subplot(122); plt.bar(x, success_rate_arr); plt.title('success percentage')
    plt.savefig('/home/abba/Desktop/success_vs_start_1000.png')

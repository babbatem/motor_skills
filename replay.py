import numpy as np
import gym
import pickle
import torch

from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.quadratic_baseline import QuadraticBaseline
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.dapg import DAPG

def get_env(env):

	# # get the correct env behavior
	if type(env) == str:
		env = GymEnv(env, vis=True)
	# elif isinstance(env, GymEnv):
	#     env = env
	# elif callable(env):
	#     env = env(**env_kwargs)
	# else:
	#     print("Unsupported environment format")
	#     raise AttributeError
	# env=gym.make(env, vis=True)
	# env.horizon=2000

	return env

def do_replays(
		num_traj,
		env,
		policy,
		eval_mode = False,
		horizon = 1e6,
		base_seed = 123,
		env_kwargs=None,
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

	env_made=get_env(env)

	if base_seed is not None:
		env_made.set_seed(base_seed)
		np.random.seed(base_seed)
	else:
		np.random.seed()
	horizon = min(horizon, env_made.horizon)
	paths = []

	for ep in range(num_traj):

		del env_made
		env_made=get_env(env)

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

				np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
				print(a[:6])

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

			# print(r)

		print('Episode Total: ', sum(rewards))

	del(env_made)
	return paths


from motor_skills.envs.mj_jaco import MjJacoDoorImpedanceCIP

# env=MjJacoDoorImpedanceCIP(vis=True)
# f=open('experiments/cip/dev/policy_15.pickle','rb')
f=open('experiments/cip/dev/policy_95.pickle','rb')
# f=open('experiments/naive/dev/policy_95.pickle','rb')
policy = pickle.load(f)
num_traj = 100

env='motor_skills:mj_jaco_door_cip-v0'
# env='motor_skills:mj_jaco_door_naive-v0'
do_replays(num_traj,
		   env,
		   policy,
		   eval_mode = True,
		   horizon = 1e6,
		   base_seed = 1
		   )

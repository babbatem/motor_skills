import time
import copy
import pathlib
import pickle
import hjson
import gym
import numpy as np
from scipy.spatial.transform import Rotation as R

from mujoco_py import cymj
from mujoco_py import load_model_from_path, MjSim, MjViewer
import motor_skills.core.mj_control as mjc
from motor_skills.cip.mppiExecutorClass import mppiExecutor
from motor_skills.cip.mppiPlannerClass import mppiPlanner


class MjJacoMPPI(gym.Env):
	""" """

	def __init__(self, vis=False, n_steps=int(2000), start=[0, 0, 0, 0, 0, 0], goal_pos=[0,0,0], goal_quat=[1, 0, 0, 0]):

		# %% setup MuJoCo
		self.parent_dir_path = str(pathlib.Path(__file__).parent.absolute())
		self.vis=vis
		self.fname = self.parent_dir_path + '/assets/kinova_j2s6s300/mj-j2s6s300_door.xml'
		self.model = load_model_from_path(self.fname)
		self.sim = MjSim(self.model)
		self.viewer = MjViewer(self.sim) if self.vis else None

		# %% load the CIP
		controller_file = self.parent_dir_path + '/assets/controller_config.hjson'
		self.cip = mppiExecutor(controller_file, self.sim)
		self.control_timestep = 1.0 / self.cip.controller.control_freq
		self.model_timestep = self.sim.model.opt.timestep
		self.arm_dof = self.cip.controller.control_dim

		# %% disable joint limits?
		for i in range(6):
			self.model.jnt_limited[i]=0

		# %% configure action space (no gripper in this case)
		self.action_dim = 6
		a_low = np.full(self.action_dim, -float('inf'))
		a_high = np.full(self.action_dim, float('inf'))
		self.action_space = gym.spaces.Box(a_low,a_high)

		self.observation_dim = self.arm_dof
		o_low = np.full(self.observation_dim, -float('inf'))
		o_high = np.full(self.observation_dim, float('inf'))
		self.observation_space=gym.spaces.Box(o_low,o_high)

		self.env=self
		self.n_steps = n_steps

		# %% start is a qpos
		self.start = start

		# %% goal in cartesian space
		self.goal_pos = goal_pos
		self.goal_quat = goal_quat # (mujoco format, scalar first)
		self.goal_rotmat = R.from_quat(mjc.quat_to_scipy(goal_quat))
		self.goal_rotmat = self.goal_rotmat.as_dcm()
		self.grp_idx = cymj._mj_name2id(self.sim.model, 1, "j2s6s300_link_6")

	def reset_model(self):
		self.sim.data.qpos[:6] = self.start
		self.sim.data.qpos[-2] = 0.0
		self.sim.data.qpos[-1] = 0.0
		self.sim.step()

	def set_seed(self, seed):
		np.random.seed(seed)

	def reset(self, seed=0):
		self.reset_model()

		# %% reset controller
		self.cip.controller.reset()
		obs = np.array(self.sim.data.qpos[:self.arm_dof])
		self.env_timestep=0
		return obs

	def cost(self):

		gripper_target_displacement = np.abs(self.sim.data.body_xpos[self.grp_idx] - self.goal_pos)
		gripper_target_distance = np.linalg.norm(gripper_target_displacement)

		# A = 1; B = 10;
		# cost = A * gripper_target_distance + B * orientation_norm
		# cost = gripper_target_distance
		l1_dist = np.sum(np.abs(gripper_target_displacement))
		l2_dist = gripper_target_distance
		position_cost = l1_dist + 5.0 * l2_dist

		current_rotmat = R.from_quat(mjc.quat_to_scipy(self.sim.data.body_xquat[self.grp_idx]))
		current_rotmat = current_rotmat.as_dcm()
		orientation_error = self.cip.controller.calculate_orientation_error(self.goal_rotmat, current_rotmat)
		orientation_norm = np.linalg.norm(orientation_error)

		# cost = position_cost + 100.0 * orientation_norm
		cost=position_cost
		return cost


	def step(self, action):

		# 	# %% interpret as torques here (gravity comp done in the CIP)

		policy_step = True
		for i in range(int(self.control_timestep / self.model_timestep)):#
			torques = self.cip.get_action(action, policy_step)
			self.sim.data.ctrl[:self.arm_dof] = torques
			self.sim.step()
			policy_step = False
			self.env_timestep+=1

		# %% if above loop is commented out:
		# torques = self.cip.get_action(action, policy_step)
		# self.sim.data.ctrl[:self.arm_dof] = torques
		# self.sim.step()
		# policy_step = False
		# self.env_timestep+=1

		self.viewer.render() if self.vis else None

		reward = -1*self.cost()
		info={'goal_achieved': reward > -1e-1 }
		done = self.env_timestep >= (self.n_steps - 1)
		obs = self.sim.data.qpos[:self.arm_dof]
		return obs, reward, done, info

	def render(self):
		try:
			self.viewer.render()
		except Exception as e:
			self.viewer = MjViewer(self.sim)
			self.viewer.render()

	def get_env_state(self):
		target_pos=self.goal_pos
		target_rot=self.goal_quat

		return dict(qp=self.sim.data.qpos.copy(), qv=self.sim.data.qvel.copy(),
					qa=self.sim.data.qacc.copy(),
					target_pos=target_pos, target_rot=target_rot,
					timestep=self.env_timestep)

	def set_env_state(self, state):
		self.sim.reset()
		qp = state['qp'].copy()
		qv = state['qv'].copy()
		qa = state['qa'].copy()
		target_pos = state['target_pos']
		self.env_timestep = state['timestep']
		# self.model.site_pos[self.target_sid] = target_pos
		self.sim.forward()
		self.sim.data.qpos[:] = qp
		self.sim.data.qvel[:] = qv
		self.sim.data.qacc[:] = qa
		self.sim.forward()

	def get_obs(self):
		return np.concatenate([
			self.sim.data.qpos.flat,
			self.sim.data.qvel.flat,
		])

	def get_env_infos(self):
		return dict(state=self.get_env_state())

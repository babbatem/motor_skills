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

		# %% configure action space (no gripper in this case)
		action_dim = self.arm_dof
		a_low = np.full(action_dim, -float('inf'))
		a_high = np.full(action_dim, float('inf'))
		self.action_space = gym.spaces.Box(a_low,a_high)

		obs_space = self.arm_dof
		o_low = np.full(obs_space, -float('inf'))
		o_high = np.full(obs_space, float('inf'))
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

	def reset(self):

		self.reset_model()

		# %% reset controller
		self.cip.controller.reset()
		obs = np.array(self.sim.data.qpos[:self.arm_dof])
		self.elapsed_steps=0
		return obs

	def cost(self):

		gripper_target_displacement = self.sim.data.body_xpos[self.grp_idx] - self.goal_pos
		gripper_target_distance = np.linalg.norm(gripper_target_displacement)

		current_rotmat = R.from_quat(mjc.quat_to_scipy(self.sim.data.body_xquat[gripper_idx]))
		current_rotmat = current_rotmat.as_dcm()
		orientation_error = self.cip.controller.calculate_orientation_error(self.goal_rotmat, current_rotmat)
		orientation_norm = np.linalg.norm(orientation_error)

		A = 1; B = 10;
		cost = A * gripper_target_distance + B * orientation_norm
		return cost


	def step(self, action):

		policy_step = True
		for i in range(int(self.control_timestep / self.model_timestep)):

			# %% interpret as torques here (gravity comp done in the CIP)
			torques = self.cip.get_action(action, policy_step)
			self.sim.data.ctrl[:self.arm_dof] = torques
			self.sim.step()
			policy_step = False
			self.elapsed_steps+=1

		self.viewer.render() if self.vis else None

		reward = -1*self.cost()
		info={'goal_achieved': reward > -1e-1 }
		done = self.elapsed_steps >= (self.n_steps - 1)
		obs = self.sim.qpos[:self.arm_dof]
		return obs, reward, done, info

	def render(self):
		try:
			self.viewer.render()
		except Exception as e:
			self.viewer = MjViewer(self.sim)
			self.viewer.render()

	def get_env_state(self):
		# %% TODO
		pass

	def set_env_state(self):
		# %% TODO
		pass

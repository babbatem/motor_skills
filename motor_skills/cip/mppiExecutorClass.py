import copy
import hjson
import numpy as np
import pickle
import gym

from motor_skills.cip.cip import CIP
from motor_skills.cip.arm_controller import PositionOrientationController
import motor_skills.core.mj_control as mjc

class mppiExecutor(CIP):
	"""docstring for MPPIPlanner."""
	def __init__(self, controller_file, sim):
		super(mppiExecutor, self).__init__()

		# %% load the controller (TODO: maybe the CIP object here)
		with open(controller_file) as f:
			params = hjson.load(f)

		params['position_orientation']['impedance_flag']=False

		self.sim = sim
		self.controller = PositionOrientationController(**params['position_orientation'])
		self.arm_dof = 6
		self.target_q = None

	def get_action(self, action, policy_step):

		# # self.controller.update_model(self.sim,
		# 							 # id_name='j2s6s300_link_6',
		# 							 # joint_index=np.arange(6))
		#
		# # torques = self.controller.action_to_torques(action,
		# 											policy_step)
		# # torques += self.sim.data.qfrc_bias[:self.arm_dof]

		# interpret as delta q at the present config.
		if policy_step:
			action=np.clip(action, -np.pi/4, np.pi/4)
			self.target_q = copy.deepcopy(self.sim.data.qpos[:6]) + action

		qdd = None
		qd  = np.zeros(6)
		q   = self.target_q
		torques = mjc.pd(qdd, qd, q, self.sim, ndof=6)
		return torques

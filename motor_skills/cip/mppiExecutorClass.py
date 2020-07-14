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

		with open(controller_file) as f:
			params = hjson.load(f)
		self.controller = PositionOrientationController(**params['position_orientation'])

		# %% load the controller (TODO: maybe the CIP object here)
		with open(controller_file) as f:
			params = hjson.load(f)

		self.sim = sim
		self.controller = PositionOrientationController(**params['position_orientation'])
		self.arm_dof = 6

	def get_action(self, action, policy_step):

		arm_action = action[:self.arm_dof]
		self.controller.update_model(self.sim,
									 id_name='j2s6s300_link_6',
									 joint_index=np.arange(6))

		torques = self.controller.action_to_torques(arm_action,
													policy_step)
		torques += self.sim.data.qfrc_bias[:self.arm_dof]
		return torques

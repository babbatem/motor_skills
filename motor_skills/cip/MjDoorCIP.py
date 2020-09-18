import copy

import pickle
import pathlib
import numpy as np

import motor_skills.core.mj_control as mjc
from motor_skills.cip.ImpedanceCIP import ImpedanceCIP
from motor_skills.cip.MjGraspHead import MjGraspHead
from motor_skills.envs.mj_jaco import mj_cip_utils as utils

parent_dir_path = str(pathlib.Path(__file__).parent.absolute())
GPD_POSES_PATH = parent_dir_path + "/../envs/mj_jaco/assets/MjJacoDoorGrasps"

class MjDoorCIP(ImpedanceCIP):
	"""
		Implements the particular CIP for solving MjJacoDoorImpedanceCIPs gym environment.
		inherits from ImpedanceCIP, which implements only get_action.
		possesses a MjDoorHead object which serves as a grasping module.

		no motion generation happens.
		to reset, the agent samples a grasp pose and is teleported there.
	"""

	def __init__(self, controller_file, sim, start_idx=None, viewer=None):
		super(MjDoorCIP, self).__init__(controller_file, sim)
		grasp_file = open(GPD_POSES_PATH, 'rb')
		self.grasp_qs = pickle.load(grasp_file)
		self.head = MjGraspHead(self.sim, viewer=viewer)

		self.sim = self.sim
		self.start_idx=start_idx

	def success_predicate(self):
		return utils.door_open_success(self.sim)

	def learning_cost(self, sim):
		return utils.dense_open_cost(sim)

	def execute_head(self):
		self.head.execute(self.sim)

	def sample_init_set(self):

		if self.start_idx is not None:
			idx = self.start_idx
		else:
			idx = np.random.randint(len(self.grasp_qs))

		# TODO: these are in joint space, eventually want ee pose.
		g = self.grasp_qs[idx]
		return g, idx

	def learning_reset(self):

		# % sample a state from init set
		grasp_config, idx = self.sample_init_set()

		# % set qpos to that state (e.g. assume perfect execution)
		self.sim.data.qpos[:6] = grasp_config
		self.sim.data.qvel[:6] = [0.0]*6

		# % open gripper
		self.sim.data.qpos[6:12] = [0.0]*6
		self.sim.data.qvel[6:12] = [0.0]*6

		# % compute a torque command to stabilize about this point.
		full_qpos = copy.deepcopy(self.sim.data.qpos[:12])
		torque = mjc.pd(None, [0.0]*12, full_qpos, self.sim, ndof=12, kp=np.eye(12)*300, kv=np.eye(12)*500)
		self.sim.data.ctrl[:]=torque
		self.sim.forward()

		# % execute the grasp
		self.execute_head()
		return idx

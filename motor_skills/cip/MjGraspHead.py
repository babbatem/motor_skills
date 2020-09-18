import time, copy

import pathlib
import pickle
import numpy as np
from mujoco_py import cymj, MjViewer
from scipy.spatial.transform import Rotation as R

import motor_skills.core.mj_control as mjc

ARMDOF=6
GRIPPERDOF=6
DOF=ARMDOF+GRIPPERDOF

PREGRASP_STEPS = 500
PREGRASP_GOAL = [0, -0.00, -0.04]
GRASP_STEPS = 500
MAX_FINGER_DELTA=1.3

parent_dir_path = str(pathlib.Path(__file__).parent.absolute())
GPD_POSES_PATH = parent_dir_path + "/../envs/mj_jaco/assets/MjJacoDoorGrasps"

class MjGraspHead(object):
	"""
		executes a grasp in MuJoCo with the kinova j2s6s300
	"""

	def __init__(self, sim, viewer=None):
		super(MjGraspHead, self).__init__()

		self.sim = sim
		self.viewer = viewer

		# % compute indices and per timestep delta q
		self.delta = np.zeros(DOF)
		self.finger_joint_idxs = []
		for i in range(1,4):
			base_idx = cymj._mj_name2id(self.sim.model, 3,"j2s6s300_joint_finger_" + str(i))
			tip_idx = cymj._mj_name2id(self.sim.model, 3,"j2s6s300_joint_finger_tip_" + str(i))
			self.finger_joint_idxs.append(base_idx)
			self.finger_joint_idxs.append(tip_idx)
			self.delta[base_idx] = MAX_FINGER_DELTA/GRASP_STEPS
			self.delta[tip_idx] = MAX_FINGER_DELTA/GRASP_STEPS

	def pregrasp(self, sim):
		"""
			approaches object.
			Moves to PREGRASP_GOAL in ee coordinates with constant orientation.
		"""

		# % compute ee pose
		obj_type = 1 # 3 for joint, 1 for body
		body_idx = cymj._mj_name2id(self.sim.model, obj_type,"j2s6s300_link_6")

		ee_frame_goal_homog = np.append(PREGRASP_GOAL, 1)

		cur_quat = copy.deepcopy(self.sim.data.body_xquat[body_idx])
		rot_mat = R.from_quat([cur_quat[1],
								cur_quat[2],
								cur_quat[3],
								cur_quat[0]])
		trans_mat = np.zeros([4,4])
		trans_mat[:3,:3] = rot_mat.as_dcm()
		trans_mat[3,:3] = 0
		trans_mat[3,3] = 1
		trans_mat[:3,3] = self.sim.data.body_xpos[body_idx]
		world_goal = np.matmul(trans_mat, ee_frame_goal_homog)[:3]

		for t in range(PREGRASP_STEPS):
			self.sim.data.ctrl[:] = mjc.ee_reg2(world_goal,
												self.sim.data.body_xquat[body_idx],
												self.sim,
												body_idx,
												kp=np.eye(3)*300, kv=None, ndof=12)
			self.sim.forward()
			self.sim.step()
			if self.viewer is not None:
				self.viewer.render()

	def execute(self, sim):
		"""
			implements a naive grasping strategy.
			closes fingers until they make contact, at which point they stop.
			this by no means ensures Grasp Stability, but it ought to be okay for grabbing cylinders.
		"""

		# % approach
		self.pregrasp(sim)

		# % reset the door
		# % reset door
		for i in range(2):
			sim.data.qpos[DOF+i]=0.0

		# % close fingers
		new_pos = copy.deepcopy(self.sim.data.qpos[:DOF])
		for t in range(GRASP_STEPS):
			new_pos += self.delta

			# % see which sensors are reporting force
			touched = np.where(self.sim.data.sensordata[:6] != 0.0)[0]

			# % if they are all in contact, we're done
			if len(touched) == 6:
				break

			# % otherwise, compute new setpoints for those which are not in contact
			current_pos = self.sim.data.qpos
			for touch_point in touched:
				new_pos[self.finger_joint_idxs[touch_point]] = current_pos[self.finger_joint_idxs[touch_point]]

			# % compute torque and step
			self.sim.data.ctrl[:] = mjc.pd([0] * DOF, [0] * DOF, new_pos, self.sim, ndof=DOF, kp=np.eye(DOF)*300)
			self.sim.forward()
			self.sim.step()

			if self.viewer is not None:
				self.viewer.render()

def seed_properly(seed_value=123):

	import os
	os.environ['PYTHONHASHSEED']=str(seed_value)

	import random
	random.seed(seed_value)

	import numpy as np
	np.random.seed(seed_value)

def plan_and_grasp_test():
	from motor_skills.envs.mj_jaco import MjJacoDoorImpedanceCIP
	from motor_skills.cip.pbplannerWrapper import pbplannerWrapper

	seed = 122
	while True:
		seed+=1
		seed_properly(seed)

		debug = True

		mp = pbplannerWrapper(debug=True)
		grasp_file = open(GPD_POSES_PATH, 'rb')
		grasp_qs = pickle.load(grasp_file)

		env = MjJacoDoorImpedanceCIP(vis=True)
		env.reset()

		# random state sample
		s = mp.planner.validityChecker.sample_state()
		env.sim.data.qpos[:6] = s
		env.sim.step()

		# random grasp candidate
		idx = np.random.randint(len(grasp_qs))
		g = grasp_qs[idx]
		mp.plan(s, g)
		mp.execute(env)

		# % reset door
		for i in range(2):
			env.sim.data.qpos[DOF+i]=0.0

		# TODO: make sure we got there

		# % grasp
		head = MjGraspHead(env.sim, debug=True)
		head.execute(env.sim)

		# % hover at ee pose after.
		obj_type = 1 # 3 for joint, 1 for body
		body_idx = cymj._mj_name2id(env.sim.model, obj_type, "j2s6s300_link_6")
		xpos = env.sim.data.body_xpos[body_idx]
		xquat = env.sim.data.body_xquat[body_idx]
		for t in range(100):

			env.sim.data.ctrl[:] = mjc.ee_reg2(xpos,
											   xquat,
											   env.sim,
											   body_idx,
											   kp=np.eye(3)*300, kv=None, ndof=12)
			env.sim.step()

			if debug:
				env.render()

def grasp_only_test():
	from motor_skills.envs.mj_jaco import MjJacoDoor

	seed = 122
	while True:
		seed+=1
		seed_properly(seed)

		debug = True

		grasp_file = open(GPD_POSES_PATH, 'rb')
		grasp_qs = pickle.load(grasp_file)

		env = MjJacoDoor(vis=True)
		env.reset()

		idx = np.random.randint(len(grasp_qs))
		s = grasp_qs[idx]
		env.sim.data.qpos[:6] = s
		env.sim.data.qpos[6:12] = [0.0]*6
		env.sim.data.qvel[:6]=[0.0]*6

		full_qpos = copy.deepcopy(env.sim.data.qpos[:12])
		torque = mjc.pd(None, [0.0]*12, full_qpos, env.sim, ndof=12, kp=np.eye(12)*300)
		env.sim.data.ctrl[:]=torque
		env.sim.step()
		env.render()

		time.sleep(2.0)

		# % reset door
		for i in range(2):
			env.sim.data.qpos[DOF+i]=0.0

		# % grasp
		head = MjGraspHead(env.sim, debug=True)
		head.execute(env.sim)

		# % hover at ee pose after.
		obj_type = 1 # 3 for joint, 1 for body
		body_idx = cymj._mj_name2id(env.sim.model, obj_type, "j2s6s300_link_6")
		xpos = env.sim.data.body_xpos[body_idx]
		xquat = env.sim.data.body_xquat[body_idx]
		for t in range(100):

			env.sim.data.ctrl[:] = mjc.ee_reg2(xpos,
											   xquat,
											   env.sim,
											   body_idx,
											   kp=np.eye(3)*300, kv=None, ndof=12)
			env.sim.step()

			if debug:
				env.render()


if __name__ == '__main__':
	# plan_and_grasp_test()
	grasp_only_test()

import time
import copy
import hjson
import numpy as np
import pickle
import gym

from motor_skills.planner.pbplanner import PbPlanner
from motor_skills.cip.arm_controller import PositionOrientationController
import motor_skills.core.mj_control as mjc

MIN_PLANNER_STEPS = 100 # minimum length of the plan
EXTRA_PLANNER_STEPS = 100 # steps to let the arm converge after executing planned trajectory
ARMDOF = 6
GRIPPERDOF = 6


class pbplannerWrapper(object):
	"""
		takes the pbplanner, purely pybullet, and provides basic trajectory
		execution in mujoco.
	"""
	def __init__(self):
		super(pbplannerWrapper, self).__init__()

		# %% load the controller (TODO: maybe the CIP object here)
		# with open(controller_file) as f:
		# 	params = hjson.load(f)
		#
		# params['position_orientation']['impedance_flag']=False
		#
		# self.sim = si
		# self.controller = PositionOrientationController(**params['position_orientation'])
		self.target_q = None
		self.target_qd = None

		# %% load that planner
		self.planner = PbPlanner()

	def plan(self, s, g):
		"""
		s and g ought to be collision free 6DoF poses
		drives mujoco arm to state g.
		and s ought to be always be sim.data.qpos[:6]??

		sets self.target_q and self.target_qd
		"""
		# s = planner.validityChecker.sample_state()
		# g = planner.validityChecker.sample_state()
		result=self.planner.plan(s, g)

		# % TODO: finish me
		result.interpolate(MIN_PLANNER_STEPS)
		H = result.getStateCount()
		self.target_q = []
		for i in range(H):
			tmp=[]
			state_t = result.getState(i)
			for j in range(ARMDOF):
				tmp.append(state_t[j])
			self.target_q.append(tmp)

		self.target_q = np.array(self.target_q)
		# self.target_qd = np.diff(self.target_q, axis=0)
		# self.target_qd = np.append(self.target_qd, np.zeros((1,6)), axis=0)
		self.target_qd = np.zeros_like(self.target_q)

		print('----')
		print(self.target_q.shape)
		print(self.target_qd.shape)


	def execute(self, sim):
		# TODO: control frequency?

		# % execute self.target_q
		H = len(self.target_q)
		for i in range(H):
			torques=mjc.pd(None, self.target_qd[i], self.target_q[i], sim,
						   ndof=6, kp=np.eye(6)*300)
			sim.data.ctrl[:6]=torques
			sim.step()
			env.viewer.render()
			time.sleep(0.1)

		# % let arm converge. 
		for j in range(EXTRA_PLANNER_STEPS):
			torques=mjc.pd(None, self.target_qd[i], self.target_q[i], sim,
						   ndof=6, kp=np.eye(6)*300)
			sim.data.ctrl[:6]=torques
			sim.step()
			env.viewer.render()
			time.sleep(0.1)


if __name__ == '__main__':

	from motor_skills.envs.mj_jaco import MjJacoDoorImpedance

	wrap = pbplannerWrapper()
	g = wrap.planner.validityChecker.sample_state()

	env = MjJacoDoorImpedance(vis=True)
	env.reset()

	s = copy.deepcopy(env.sim.data.qpos[:6])
	wrap.plan(s, g)
	wrap.execute(env.sim)

	#
	# while True:
	# 	env.sim.data.ctrl[:ARMDOF+GRIPPERDOF] = env.sim.data.qfrc_bias[:ARMDOF+GRIPPERDOF]
	# 	env.sim.step()
	# 	env.render()

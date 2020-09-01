import time
import copy
import hjson
import numpy as np
import pickle
import gym

from motor_skills.planner.pbplanner import PbPlanner
from motor_skills.cip.arm_controller import PositionOrientationController
import motor_skills.core.mj_control as mjc

MIN_PLANNER_STEPS = 1000 # minimum length of the plan
EXTRA_PLANNER_STEPS = 200 # steps to let the arm converge after executing planned trajectory
ARMDOF = 6
GRIPPERDOF = 6


class pbplannerWrapper(object):
	"""
		takes the pbplanner, purely pybullet, and provides basic trajectory
		execution in mujoco.
	"""
	def __init__(self, debug=False):
		self.planner = PbPlanner()
		self.target_q = None
		self.target_qd = None
		self.debug=debug


	def plan(self, s, g):
		"""
		s and g ought to be collision free 6DoF poses
		drives mujoco arm to state g.
		and s ought to be always be sim.data.qpos[:6]??

		sets self.target_q and self.target_qd
		"""
		result=self.planner.plan(s, g)
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
		self.target_qd = np.zeros_like(self.target_q)

	def execute(self, env):
		# TODO: control frequency?

		# % execute self.target_q waypoints
		H = len(self.target_q)
		for i in range(H):
			torques=mjc.pd(None, self.target_qd[i], self.target_q[i], env.sim,
						   ndof=6, kp=np.eye(6)*300)
			env.sim.data.ctrl[:6]=torques
			env.sim.step()

			if self.debug:
				env.viewer.render()
				# time.sleep(0.1)
		# % let arm converge.
		for j in range(EXTRA_PLANNER_STEPS):
			torques=mjc.pd(None, self.target_qd[i], self.target_q[i], env.sim,
						   ndof=6, kp=np.eye(6)*300)
			env.sim.data.ctrl[:6]=torques
			env.sim.step()

			if self.debug:
				env.viewer.render()
				# time.sleep(0.1)


if __name__ == '__main__':

	from motor_skills.envs.mj_jaco import MjJacoDoorImpedanceNaive

	wrap = pbplannerWrapper(debug=True)
	g = wrap.planner.validityChecker.sample_state()
	s = wrap.planner.validityChecker.sample_state()

	env = MjJacoDoorImpedanceNaive(vis=True)
	env.reset()

	env.sim.data.qpos[:6]=s
	torques=mjc.pd(None, [0.0]*6, s, env.sim,
				   ndof=6, kp=np.eye(6)*300, kv=np.eye(6)*150)
	env.sim.data.ctrl[:6]=torques
	env.sim.step()
	env.render()

	wrap.plan(s, g)
	wrap.execute(env)

	#
	# while True:
	# 	env.sim.data.ctrl[:ARMDOF+GRIPPERDOF] = env.sim.data.qfrc_bias[:ARMDOF+GRIPPERDOF]
	# 	env.sim.step()
	# 	env.render()

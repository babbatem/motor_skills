import os
import random

import numpy as np
from mujoco_py import cymj

DOOR_GOAL_OPEN = np.pi / 2
HANDLE_GOAL_OPEN = np.pi / 2
DOOR_GOAL_CLOSED = 0.0
HANDLE_GOAL_CLOSED = 0.0
EPSILON = 1e-2
DOOR_WEIGHT = 100
HANDLE_WEIGHT = 10
GRASP_WEIGHT = 0

def sample_random_pose(sim, model):

	# %% sample random start in [-pi,pi], make sure it is valid.
	random_q = 2 * np.pi * np.random.rand(6) - np.pi
	for i in range(6):
		if not model.jnt_limited[i]:
			sim.data.qpos[i]=random_q[i]
		else:
			if (random_q[i] >= model.jnt_range[i][0]):
				if (random_q[i] <= model.jnt_range[i][1]):
					sim.data.qpos[i]=random_q[i]
				else:
					sim.data.qpos[i]=model.jnt_range[i][1]
			else:
				sim.data.qpos[i]=model.jnt_range[i][0]

	sim.step()


def door_open_success(sim):
	return sim.data.qpos[-2] >= (DOOR_GOAL_OPEN - EPSILON)

def door_closed_success(sim):
	return sim.data.qpos[-2] <= (DOOR_GOAL_CLOSED + EPSILON)

def dense_open_cost(sim):
	gripper_idx = cymj._mj_name2id(sim.model, 1, "j2s6s300_link_6")
	handle_idx  = cymj._mj_name2id(sim.model, 1, "latch")
	gripper_handle_displacement = np.array(sim.data.body_xpos[gripper_idx]) - \
								  np.array(sim.data.body_xpos[handle_idx])

	gripper_handle_distance = np.linalg.norm(gripper_handle_displacement)
	# cost = DOOR_WEIGHT*(DOOR_GOAL_OPEN - max(sim.data.qpos[-2], 0) )**2 + \
	#      HANDLE_WEIGHT*(HANDLE_GOAL_OPEN - sim.data.qpos[-1])**2 + \
	#      GRASP_WEIGHT*(gripper_handle_distance)  # note: I added the max( . , 0) to prevent penalizing for pushing the door forward.

	cost = DOOR_WEIGHT*(DOOR_GOAL_OPEN - sim.data.qpos[-2])**2 + \
		   HANDLE_WEIGHT*(HANDLE_GOAL_OPEN - sim.data.qpos[-1])**2 + \
		   GRASP_WEIGHT*(gripper_handle_distance)  # note: I added the max( . , 0) to prevent penalizing for pushing the door forward.
	return cost

def dense_closing_cost(sim):
	gripper_idx = cymj._mj_name2id(sim.model, 1, "j2s6s300_link_6")
	handle_idx  = cymj._mj_name2id(sim.model, 1, "latch")
	gripper_handle_displacement = sim.data.body_xpos[gripper_idx] - \
					sim.data.body_xpos[handle_idx]

	gripper_handle_distance = np.linalg.norm(gripper_handle_displacement)
	cost = DOOR_WEIGHT*(DOOR_GOAL_ - sim.data.qpos[-2])**2 + \
		 HANDLE_WEIGHT*(HANDLE_GOAL_CLOSED - sim.data.qpos[-1])**2 + \
		 GRASP_WEIGHT*(gripper_handle_distance)
	return cost

def seed_properly(seed_value):
	os.environ['PYTHONHASHSEED']=str(seed_value)
	random.seed(seed_value)
	np.random.seed(seed_value)

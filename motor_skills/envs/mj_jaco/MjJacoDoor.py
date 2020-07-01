import time
import copy
import pathlib
import pickle
import gym
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
from motor_skills.envs.mj_jaco import MjJacoEnv
import motor_skills.core.mj_control as mjc
from mujoco_py import cymj
from scipy.spatial.transform import Rotation as R

THRESHOLD = np.pi / 2

class MjJacoDoor(gym.Env):
    """docstring for MjJacoDoor."""

    def __init__(self, vis=False, n_steps=int(1000)):
        self.parent_dir_path = str(pathlib.Path(__file__).parent.absolute())
        self.vis=vis
        self.fname = self.parent_dir_path + '/assets/kinova_j2s6s300/mj-j2s6s300_door.xml'
        self.model = load_model_from_path(self.fname)
        self.sim = MjSim(self.model)
        self.viewer = MjViewer(self.sim) if self.vis else None

        a_low = np.full(12, -float('inf'))
        a_high = np.full(12, float('inf'))
        self.action_space = gym.spaces.Box(a_low,a_high)

        obs_space = self.model.nq + self.model.nsensordata
        o_low = np.full(obs_space, -float('inf'))
        o_high = np.full(obs_space, float('inf'))
        self.observation_space=gym.spaces.Box(o_low,o_high)
        self.env=self
        self.n_steps = n_steps


    def reset(self):

        # pick a random start arm pose (DoFs 1-6)
        start_pose_file = open(self.parent_dir_path + "/assets/MjJacoDoorGrasps", 'rb')
        self.start_poses = pickle.load(start_pose_file)
        idx = np.random.randint(len(self.start_poses))
        # self.sim.data.qpos[:6]=self.start_poses[idx]
        self.sim.data.qpos[:6] = self.start_poses[8]
        self.sim.step()

        # TODO: close the gripper here
        # a heuristic strategy: close fingers until the first link is in contact.
        # then close finger tips in the same fashion

        # ee_index = 6
        # old = self.sim.data.body_xpos[ee_index].copy()
        # qpos_offset = np.zeros(12)
        # qpos_offset[:6] = [-0.123224540, .121126694, .129906782, -0.0438253357, -0.00798207077, -0.0000539686959]
        # qpos_goal = self.sim.data.qpos[:12] + qpos_offset
        # global_goal = np.zeros(6)
        obj_type = 1 # 3 for joint, 1 for body
        body_idx = cymj._mj_name2id(self.sim.model, obj_type,"j2s6s300_link_6")
        ee_frame_goal = [0.03461422, 0.02575592, -0.00387646] + self.sim.data.body_xpos[body_idx]
        ee_frame_goal = np.append(ee_frame_goal, 1)
        # print(self.sim.data.body_xquat[body_idx].shape)
        rot_mat = R.from_quat(self.sim.data.body_xquat)
        trans_mat = np.zeros([4,4])
        trans_mat[:3,:3] = rot_mat.as_dcm()[body_idx]
        trans_mat[3,:3] = 0
        trans_mat[3,3] = 1
        trans_mat[:3,3] = self.sim.data.body_xpos[body_idx]
        world_goal = np.matmul(trans_mat, ee_frame_goal)[:3]

        # print(self.sim.data.body_xpos[body_idx])
        # print(world_goal)
        # print(global_goal)
        for t in range(1000):
            self.sim.data.ctrl[:] = mjc.ee_regulation(world_goal, self.sim, body_idx, kp=None, kv=None, ndof=12)
            self.sim.forward()
            self.sim.step()
            self.viewer.render()


        # print('Diff: ' + str(self.sim.data.body_xpos[ee_index] - old))
        offset = np.zeros(12)
        for i in range(1,4):
            base_idx = cymj._mj_name2id(self.sim.model, obj_type,"j2s6s300_joint_finger_" + str(i))
            tip_idx = cymj._mj_name2id(self.sim.model, obj_type,"j2s6s300_joint_finger_tip_" + str(i))
            offset[base_idx] = 2
            offset[tip_idx] = 2
            new_pos = self.sim.data.qpos[:12] + offset

        self.sim.data.ctrl[:] = mjc.pd([0] * 12, [0] * 12, new_pos, self.sim, ndof=12)

        for t in range(5000):
            # print('Goal: ' + str(new_pos))
            # print('Current: ' + str(self.sim.data.qpos))
            self.sim.forward()
            self.sim.step()
            self.viewer.render()
            print(self.sim.data.sensordata)
            touched = np.where(self.sim.data.sensordata[:6] != 0.0)
            # print(touched)
            if len(touched) == len(self.sim.data.sensordata):
                break
            current_pos = self.sim.data.qpos
            print(current_pos)
            # for touch_point in touched:
            #     new_pos[touch_point] = current_pos[touch_point]
            self.sim.data.ctrl[:] = mjc.pd([0] * 12, [0] * 12, new_pos, self.sim, ndof=12)

        # reset the object
        self.sim.data.qpos[-1]=0.0
        self.sim.data.qpos[-2]=0.0

        self.elapsed_steps=0
        obs = np.concatenate([self.sim.data.qpos, self.sim.data.sensordata])
        return obs

    def step(self, action):

        # TODO: failure predicate here
        # if contact is lost for some number of timesteps, exit and return -1
        # this might lead to a policy that doesn't do anything if reward is too sparse
        # we ought to give this some more thought.

        for i in range(len(action)):
            self.sim.data.ctrl[i]=action[i]+self.sim.data.qfrc_bias[i]

        self.sim.forward()
        self.sim.step()
        self.viewer.render() if self.vis else None

        reward = self.sim.data.qpos[-2] > THRESHOLD

        info={'goal_achieved': reward}

        done = self.elapsed_steps == (self.n_steps - 1)

        obs = np.concatenate([self.sim.data.qpos, self.sim.data.sensordata])

        self.elapsed_steps+=1
        return obs, reward, done, info

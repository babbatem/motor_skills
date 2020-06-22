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

        ee_index = 6
        ee_xpos_goal = copy.deepcopy(self.sim.data.body_xpos[ee_index])
        ee_xpos_goal[1] += 0.2
        ee_xpos_goal[2] -= 0.005

        for t in range(1000):
            self.sim.data.ctrl[:] = mjc.ee_regulation(ee_xpos_goal, self.sim, ee_index, kp=None, kv=None, ndof=12)
            self.sim.step()
            self.sim.forward()
            self.viewer.render()

        print(self.sim.data.qpos)
        obj_type = 3 # 3 for joint, 1 for body
        joint_idxs = np.array([])
        for i in range(1,4):
            base_idx = cymj._mj_name2id(self.sim.model, obj_type,"j2s6s300_joint_finger_" + str(i))
            tip_idx = cymj._mj_name2id(self.sim.model, obj_type,"j2s6s300_joint_finger_tip_" + str(i))
            offset = np.zeros(12)
            offset[base_idx] = 2
            offset[tip_idx] = 2
            new_pos = self.sim.data.qpos[:12] + offset
            for t in range(100):
                self.sim.data.ctrl[:] = mjc.pd([0] * 12, [0] * 12, new_pos, self.sim, ndof=12)
                self.sim.step()
                self.sim.forward()
                self.viewer.render()
                print(self.sim.data.sensordata)
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

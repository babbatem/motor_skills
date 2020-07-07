import time
import copy
import pathlib
import pickle
import hjson
import gym
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
from motor_skills.envs.mj_jaco import MjJacoEnv
import motor_skills.core.mj_control as mjc
from mujoco_py import cymj
from scipy.spatial.transform import Rotation as R

from motor_skills.cip.ImpedanceCIP import ImpedanceCIP

DOOR_GOAL = np.pi / 2

class MjJacoDoorImpedance(gym.Env):
    """docstring for MjJacoDoor."""

    def __init__(self, vis=True, n_steps=int(20000)):

        # %% setup MuJoCo
        self.parent_dir_path = str(pathlib.Path(__file__).parent.absolute())
        self.vis=vis
        self.fname = self.parent_dir_path + '/assets/kinova_j2s6s300/mj-j2s6s300_door.xml'
        self.model = load_model_from_path(self.fname)
        self.sim = MjSim(self.model)
        self.viewer = MjViewer(self.sim) if self.vis else None

        # %% load the CIP
        controller_file = self.parent_dir_path + '/assets/controller_config.hjson'
        self.cip = ImpedanceCIP(controller_file, self.sim)
        self.control_timestep = 1.0 / self.cip.controller.control_freq
        self.model_timestep = self.sim.model.opt.timestep
        self.arm_dof = self.cip.controller.control_dim

        # %% configure action space (+6 for the gripper)
        action_dim = self.cip.controller.action_dim
        a_low = np.full(action_dim, -float('inf'))
        a_high = np.full(action_dim, float('inf'))
        self.action_space = gym.spaces.Box(a_low,a_high)

        obs_space = self.model.nq + self.model.nsensordata
        o_low = np.full(obs_space, -float('inf'))
        o_high = np.full(obs_space, float('inf'))
        self.observation_space=gym.spaces.Box(o_low,o_high)
        self.env=self
        self.n_steps = n_steps

    def reset(self):

        # %% TODO: sample start state from CIP init set.

        # %% reset controller
        self.cip.controller.reset()

        # %% reset gripper
        self.sim.data.qpos[6:12] = np.zeros(6)

        # for now:
        # pick a random start arm pose (DoFs 1-6)
        start_pose_file = open(self.parent_dir_path + "/assets/MjJacoDoorGrasps", 'rb')
        self.start_poses = pickle.load(start_pose_file)
        idx = 8 # np.random.randint(len(self.start_poses))
        self.sim.data.qpos[:6] = self.start_poses[idx]
        self.sim.step()
        obs = np.concatenate([self.sim.data.qpos, self.sim.data.sensordata])
        self.elapsed_steps=0
        return obs

    def step(self, action):

        # %% interpret action as target [pos, ori] of gripper
        policy_step = True
        for i in range(int(self.control_timestep / self.model_timestep)):
            torques = self.cip.get_action(action, policy_step)
            torques += self.sim.data.qfrc_bias[:self.arm_dof]
            self.sim.data.ctrl[:self.arm_dof] = torques

            # %% TODO: gripper action
            self.sim.step()
            policy_step = False
            self.elapsed_steps+=1

        self.sim.step()
        self.viewer.render() if self.vis else None

        reward = -1*(DOOR_GOAL - self.sim.data.qpos[-1])**2
        info={'goal_achieved': reward > -1e-1 }
        done = self.elapsed_steps >= (self.n_steps - 1)
        obs = np.concatenate([self.sim.data.qpos, self.sim.data.sensordata])
        return obs, reward, done, info

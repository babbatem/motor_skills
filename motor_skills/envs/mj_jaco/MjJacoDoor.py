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
        self.sim.data.qpos[:6]=self.start_poses[idx]
        # self.sim.data.qpos[:6] = self.start_poses[8]
        self.sim.step()

        # TODO: close the gripper here
        # a heuristic strategy: close fingers until the first link is in contact.
        # then close finger tips in the same fashion

        obj_type = 1 # 3 for joint, 1 for body
        body_idx = cymj._mj_name2id(self.sim.model, obj_type,"j2s6s300_link_6")

        ee_frame_goal = [0, -0.01, -0.04]
        ee_frame_goal = np.append(ee_frame_goal, 1)

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
        world_goal = np.matmul(trans_mat, ee_frame_goal)[:3]

        for t in range(10000):
            self.sim.data.ctrl[:] = mjc.ee_reg2(world_goal, self.sim.data.body_xquat[body_idx], self.sim, body_idx, kp=None, kv=None, ndof=12)
            self.sim.forward()
            self.sim.step()
            self.viewer.render()

        FINGER_STEPS = 5000
        new_pos = self.sim.data.qpos[:12]
        small_offset = np.zeros(12)
        finger_joint_idxs = []
        for i in range(1,4):
            base_idx = cymj._mj_name2id(self.sim.model, 3,"j2s6s300_joint_finger_" + str(i))
            tip_idx = cymj._mj_name2id(self.sim.model, 3,"j2s6s300_joint_finger_tip_" + str(i))
            finger_joint_idxs.append(base_idx)
            finger_joint_idxs.append(tip_idx)
            small_offset[base_idx] = 1.3/FINGER_STEPS
            small_offset[tip_idx] = 1.3/FINGER_STEPS

        for t in range(FINGER_STEPS):
            new_pos += small_offset
            touched = np.where(self.sim.data.sensordata[:6] != 0.0)[0]
            print(touched)
            if len(touched) == 6:
                break
            current_pos = self.sim.data.qpos
            for touch_point in touched:
                print(finger_joint_idxs[touch_point])
                new_pos[finger_joint_idxs[touch_point]] = current_pos[finger_joint_idxs[touch_point]]
            self.sim.data.ctrl[:] = mjc.pd([0] * 12, [0] * 12, new_pos, self.sim, ndof=12)
            self.sim.forward()
            self.sim.step()
            self.viewer.render()

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

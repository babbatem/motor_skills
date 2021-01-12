import hjson
import copy
import numpy as np

from motor_skills.cip.cip import CIP
from motor_skills.cip.arm_controller import PositionOrientationController
import motor_skills.core.mj_control as mjc

class ImpedanceCIP(CIP):
    """docstring for ImpedanceCIP."""
    def __init__(self, controller_file, sim):
        super(ImpedanceCIP, self).__init__()

        with open(controller_file) as f:
            params = hjson.load(f)
        self.controller = PositionOrientationController(**params['position_orientation'])

        # %% load the controller (TODO: maybe the CIP object here)
        with open(controller_file) as f:
            params = hjson.load(f)

        self.sim = sim
        self.controller = PositionOrientationController(**params['position_orientation'])
        self.arm_dof = 6
        self.grp_idx=np.arange(self.arm_dof, len(self.sim.data.ctrl))
        self.grp_target=None

    def set_gripper_target(self, target):
        self.grp_target = target

    def get_action(self, action, policy_step):
        # TODO: failure predicate here (within CIP)
        # if contact is lost for some number of timesteps, exit and return -1
        # this might lead to a policy that doesn't do anything if reward is too sparse
        # we ought to give this some more thought.

        # %% split action in arm, gripper
        arm_action = action[:self.controller.action_dim]

        self.controller.update_model(self.sim,
                                     id_name='j2s6s300_link_6',
                                     joint_index=np.arange(6))

        torques = self.controller.action_to_torques(arm_action,
                                                    policy_step)
        torques += self.sim.data.qfrc_bias[:self.arm_dof]

        # NOTE: gripper control disabled
        # %% treat gripper action as delta position (fixed kp, kv)
        # %% apply delta on policy step; otherwise, let it be
        # gripper_action = action[self.controller.action_dim:]
        # if policy_step:
        #     self.grp_target = copy.deepcopy(self.sim.data.qpos[:len(self.sim.data.ctrl)])
        #     self.grp_target[self.grp_idx]+=gripper_action / 1e2

        gripper_torques = mjc.pd(None,
                                 np.zeros(len(self.sim.data.ctrl)),
                                 self.grp_target,
                                 self.sim,
                                 kp=np.eye(12)*600,
                                 kv=np.eye(12)*300)

        gripper_torques=gripper_torques[self.grp_idx]

        # TODO: safety constraints here.
        return np.concatenate( [torques, gripper_torques] )

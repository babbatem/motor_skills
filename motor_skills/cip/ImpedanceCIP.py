import hjson
import numpy as np

from motor_skills.cip.cip import CIP
from motor_skills.cip.arm_controller import PositionOrientationController

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

    def get_action(self, action, policy_step):
        # TODO: failure predicate here (within CIP)
        # if contact is lost for some number of timesteps, exit and return -1
        # this might lead to a policy that doesn't do anything if reward is too sparse
        # we ought to give this some more thought.

        self.controller.update_model(self.sim,
                                     id_name='j2s6s300_link_6',
                                     joint_index=np.arange(6))

        torques = self.controller.action_to_torques(action,
                                                    policy_step)

        # TODO: safety constraints here.

        return torques

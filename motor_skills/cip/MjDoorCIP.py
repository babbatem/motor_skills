import numpy as np
import pickle

from motor_skills.cip.ImpedanceCIP import ImpedanceCIP
from motor_skills.cip.MjGraspHead import MjGraspHead
import motor_skills.envs.mj_jaco.mj_cip_utils as utils

GPD_POSES_PATH = "/home/abba/msu_ws/src/motor_skills/motor_skills/envs/mj_jaco/assets/MjJacoDoorGrasps"

class MjDoorCIP(ImpedanceCIP):
    """
        Implements the particular CIP for solving MjJacoDoorImpedanceCIPs gym environment.
        inherits from ImpedanceCIP, which implements only get_action.
        possesses a MjDoorHead object which serves as a grasping module.

        no motion generation happens.
        to reset, the agent samples a grasp pose and is teleported there.
    """

    def __init__(self, controller_file, sim):
        super(MjDoorCIP, self).__init__(controller_file, sim)
        grasp_file = open(GPD_POSES_PATH, 'rb')
        self.grasp_qs = pickle.load(grasp_file)
        self.head = MjGraspHead(self.sim, debug=False)

        self.sim = self.sim

    def success_predicate(self):
        return utils.door_open_success(self.sim)

    def learning_cost(self):
        return utils.dense_open_cost(self.sim)

    def execute_head(self):
        self.head.execute(self.sim)

    def sample_init_set(self):

        # TODO: these are in joint space, eventually want ee pose.
        idx = np.random.randint(len(self.grasp_qs))
		g = grasp_qs[idx]
		return g

    def learning_reset(self):

        # % sample a state from init set
        grasp_config = self.sample_init_set()

        # % set qpos to that state (e.g. assume perfect execution)
        self.sim.data.qpos[:6] = grasp_config
        self.sim.qfrc_applied[:12] = self.sim.qfrc_bias[:12]
        self.sim.step()

        # % execute the grasp
        self.execute_head()
        return

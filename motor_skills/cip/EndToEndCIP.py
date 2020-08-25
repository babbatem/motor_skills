from motor_skills.cip.ImpedanceCIP import ImpedanceCIP
import motor_skills.envs.mj_jaco.mj_cip_utils as utils

class EndToEndCIP(ImpedanceCIP):
    def __init__(self, controller_file, sim):
        """

            Implements end-to-end learner as a CIP.

            Selects success predicate, learning cost, and learning_reset.

            This agent starts in a random pose, attempts to open the door.

        """
        super(ImpedanceCIP, self).__init__(controller_file, sim)

    def success_predicate(self):
        return utils.door_open_success(self.sim)

    def learning_cost(self):
        return utils.dense_open_cost(self.sim)

    def learning_reset(self):
        utils.sample_random_pose()

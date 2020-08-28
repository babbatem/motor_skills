from motor_skills.cip.MjDoorCIP import MjDoorCIP
from motor_skills.envs.mj_jaco.MjJacoDoorCIPBase import MjJacoDoorCIPBase


class MjJacoDoorImpedanceCIP(MjJacoDoorCIPBase):
	"""
		environment for the end-to-end agent.
	"""

	def __init__(self, vis=False, n_steps=int(2000)):

		# % call super to load model and call init_cip
		super(MjJacoDoorImpedanceCIP, self).__init__(vis=vis,n_steps=n_steps)


	def init_cip(self):

		# %% load the CIP
		controller_file = self.parent_dir_path + '/assets/controller_config.hjson'
		self.cip = MjDoorCIP(controller_file, self.sim)

from motor_skills.cip.MjDoorCIP import MjDoorCIP
from motor_skills.env.mj_jaco.MjJacoDoorCIPBase import MjJacoDoorCIPBase


class MjJacoDoorImpedanceCIPs(MjJacoDoorCIPBase):
	"""
		environment for the end-to-end agent.

		TODO: success somewhere (base env? in the CIP?)
	"""

	def __init__(self, vis=False, n_steps=int(2000)):

		# % call super to load model and call init_cip
		super(MjJacoDoorImpedanceNaive, self).__init__(vis=vis,n_steps=n_steps)


	def init_cip(self):

		# %% load the CIP
		controller_file = self.parent_dir_path + '/assets/controller_config.hjson'
		self.cip = MjDoorCIP(controller_file, self.sim)

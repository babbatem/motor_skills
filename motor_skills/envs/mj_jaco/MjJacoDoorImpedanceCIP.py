from motor_skills.cip.MjDoorCIP import MjDoorCIP
from motor_skills.envs.mj_jaco.MjJacoDoorCIPBase import MjJacoDoorCIPBase


class MjJacoDoorImpedanceCIP(MjJacoDoorCIPBase):
	"""
		environment for the end-to-end agent.
	"""

	def __init__(self, vis=False, vis_head=False, n_steps=int(2000),
				 start_idx=None, sensor_type='normal', wrist_sensor=True):

		self.start_idx = start_idx
		self.vis_head = vis_head

		# % call super to load model and call init_cip
		super(MjJacoDoorImpedanceCIP, self).__init__(vis=vis,n_steps=n_steps, sensor_type=sensor_type, wrist_sensor=wrist_sensor)



	def init_cip(self):

		# %% load the CIP
		controller_file = self.parent_dir_path + '/assets/controller_config.hjson'
		self.cip = MjDoorCIP(controller_file,
							 self.sim,
							 start_idx=self.start_idx,
							 viewer=self.viewer if self.vis_head else None)

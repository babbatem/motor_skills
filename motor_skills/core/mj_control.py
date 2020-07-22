import copy
from tqdm import tqdm
import numpy as np
from scipy.spatial.transform import Rotation as R
import mujoco_py

def gravity_comp(sim, ndof=9):
	# % qfrc_bias represents sum of Coriolis and gravity forces.
	return sim.data.qfrc_bias[:ndof]

def get_mass_matrix(sim, ndof):

	# % prepare array to hold result
	m = np.ndarray(shape=(len(sim.data.qvel)**2,),
							 dtype=np.float64,
							 order='C')

	# % call mujoco internal inertia matrix fuction
	mujoco_py.cymj._mj_fullM(sim.model, m, sim.data.qM)

	# % reshape to square, slice, and return
	m=m.reshape(len(sim.data.qvel),-1)
	return m[:ndof,:ndof]

def pd(qdd, qd, q, sim, kp=None, kv=None, ndof=12):
	"""
	inputs (in joint space):
		qdd: desired accel
		qd: desired vel
		q: desire pos
		(if any are None, that term is omitted from the control law)

	kp, kv are scalars in 1D, PSD matrices otherwise
	ndof is the number of degrees of freedom of the robot

	returns M(q)[qdd + kpE + kvEd] + H(q,qdot)
	with E = (q - sim.data.qpos), Ed = (qd - sim.data.qvel), H = sim.data.qfrc_bias
	"""

	# % handle None inputs
	q = sim.data.qpos[:ndof] if q is None else q
	qd = sim.data.qvel[:ndof] if qd is None else qd
	qdd = [0]*len(sim.data.qpos[:ndof]) if qdd is None else qdd
	kp = np.eye(len(sim.data.qpos[:ndof]))*150 if kp is None else kp
	kv = np.eye(len(sim.data.qpos[:ndof]))*10 if kv is None else kv

	# % compute the control as above
	m = get_mass_matrix(sim, ndof)
	bias = sim.data.qfrc_bias[:ndof]
	e = q - sim.data.qpos[:ndof]
	ed = qd - sim.data.qvel[:ndof]
	tau_prime = qdd + np.matmul(kp, e) + np.matmul(kv, ed)
	return np.matmul(m, tau_prime) + bias

def jac(sim, body, ndof):
	"""
	Computes Jacobian of body using q = sim.data.qpos, qdot = sim.data.qvel.
	returns jacp, jacr (position, orientation jacobians)

	note: I think I will soon concatenate jacp, jacr
	"""
	jacp = np.ndarray(shape=(3*len(sim.data.qpos)),
					  dtype=np.float64,
					  order='C')

	jacr = np.ndarray(shape=jacp.shape,
					  dtype=np.float64,
					  order='C')

	mujoco_py.cymj._mj_jacBody(sim.model,
							   sim.data,
							   jacp,
							   jacr,
							   body)
	jacp=jacp.reshape(3,-1)
	jacr=jacr.reshape(3,-1)
	return jacp[:,:ndof], jacr[:,:ndof]

# def mj_ik_traj(y_star, T, env, ee_index, ndof=9):
#     """
#     moves end effector in a straight line in cartesian space from present pose to y_star
#     TODO: position only at present - add orientation
#     TODO: collision checking
#     TODO: use fake environment for planning without execution
#     """
#     sim=env.sim
#     y0 = np.array(sim.data.body_xpos[ee_index])
#     q = sim.data.qpos[:ndof]
#     qs=[]
#     qs.append(q)
#     ys = []
#
#     for t in range(1,T):
#         y=np.array(sim.data.body_xpos[ee_index])
#         q = sim.data.qpos[:ndof]
#         qvel=sim.data.qvel[:ndof]
#
#         jacp,jacr = jac(sim, ee_index, ndof)
#         y_hat = y0 + ( t*1.0 / (T*1.0) ) * (y_star-y0)
#
#         jacp=jacp.reshape(3,-1)
#
#         # % new joint positions
#         q_update = np.linalg.pinv(jacp).dot( (y_hat - y).reshape(3,1) )
#         q = q + q_update[:len(sim.data.qpos[:ndof])].reshape(len(sim.data.qpos[:ndof]),)
#         qs.append(q)
#         ys.append(y)
#         action=pd(None,None, qs[t], env.sim)
#         env.step(action)
#
#     return qs, ys

def ee_regulation(x_des, sim, ee_index, kp=None, kv=None, ndof=9):
	"""
	This is pointless at present, but it is a building block
	for more complex cartesian control.

	PD control with gravity compensation in cartesian space
	returns J^T(q)[kp(x_des - x) - kv(xdot)] + H(q,qdot)

	TODO: quaternions or axis angles for full ee pose.
	"""
	kp = np.eye(len(sim.data.body_xpos[ee_index]))*10 if kp is None else kp
	kv = np.eye(len(sim.data.body_xpos[ee_index]))*1 if kv is None else kv

	jacp,jacr=jac(sim, ee_index, ndof)

	# % compute
	xdot = np.matmul(jacp, sim.data.qvel[:ndof])
	error_vel = xdot
	error_pos = x_des - sim.data.body_xpos[ee_index]

	pos_term = np.matmul(kp,error_pos)
	vel_term = np.matmul(kv,error_vel)

	# % commanding ee pose only
	F = pos_term - vel_term
	torques = np.matmul(jacp.T, F) + sim.data.qfrc_bias[:ndof]
	# torques = np.matmul(jacp.T, F)

	return torques

def calculate_orientation_error(desired, current):
	"""
	Optimized function to determine orientation error
	borrowed from robosuite. thanks, robosuite.
	"""

	def cross_product(vec1, vec2):
		S = np.array(([0, -vec1[2], vec1[1]],
					  [vec1[2], 0, -vec1[0]],
					  [-vec1[1], vec1[0], 0]))

		return np.dot(S, vec2)

	rc1 = current[0:3, 0]
	rc2 = current[0:3, 1]
	rc3 = current[0:3, 2]
	rd1 = desired[0:3, 0]
	rd2 = desired[0:3, 1]
	rd3 = desired[0:3, 2]

	orientation_error = 0.5 * (cross_product(rc1, rd1) + cross_product(rc2, rd2) + cross_product(rc3, rd3))

	return orientation_error

def ee_reg2(x_des, quat_des, sim, ee_index, kp=None, kv=None, ndof=12):
	"""
	same as ee_regulation, but now also accepting quat_des.
	"""
	kp = np.eye(len(sim.data.body_xpos[ee_index]))*10 if kp is None else kp
	kv = np.eye(len(sim.data.body_xpos[ee_index]))*1 if kv is None else kv

	jacp,jacr=jac(sim, ee_index, ndof)

	# % compute position error terms as before
	xdot = np.matmul(jacp, sim.data.qvel[:ndof])
	error_vel = xdot
	error_pos = x_des - sim.data.body_xpos[ee_index]

	pos_term = np.matmul(kp,error_pos)
	vel_term = np.matmul(kv,error_vel)

	# % compute orientation error terms
	current_ee_quat = copy.deepcopy(sim.data.body_xquat[ee_index])
	current_ee_rotmat = R.from_quat([current_ee_quat[1],
									 current_ee_quat[2],
									 current_ee_quat[3],
									 current_ee_quat[0]])

	target_ee_rotmat = R.from_quat([quat_des[1],
									quat_des[2],
									quat_des[3],
									quat_des[0]])

	ori_error = calculate_orientation_error(target_ee_rotmat.as_dcm(), current_ee_rotmat.as_dcm())
	euler_dot = np.matmul(jacr, sim.data.qvel[:ndof])
	ori_pos_term = np.matmul(kp, ori_error)
	ori_vel_term = np.matmul(kv, euler_dot)


	# % commanding ee pose only
	F_pos = pos_term - vel_term
	F_ori = ori_pos_term - ori_vel_term
	J_full = np.concatenate([jacp, jacr])
	F_full = np.concatenate([F_pos, F_ori])

	torques = np.matmul(J_full.T, F_full) + sim.data.qfrc_bias[:ndof]
	return torques

def generate_random_goal(n=9):
	return np.random.rand(n)*np.pi / 2.0

def quat_to_scipy(q):
	""" scalar last, [x,y,z,w]"""
	return [q[1], q[2], q[3], q[0]]

def quat_to_mj(q):
	""" scalar first, [w,x,y,z]"""
	return [q[-1], q[0], q[1], q[2]]

# %%
def pseudoinv(J):
	""" J is np.ndarray """
	return J.T.dot(np.linalg.inv(J.dot(J.T)))

def ee_traj(y_star, T, sim, ee_index):
	"""
		naive straight line between present ee position and y_stars.
		TODO: orientation.
	"""
	y0 = np.array(copy.deepcopy(sim.data.body_xpos[ee_index]))
	y0_quat = np.array(copy.deepcopy(sim.data.body_xquat[ee_index]))
	y_star = np.array(y_star)
	ys=[]
	for t in range(T):
		# % new ee pose
		# % TODO: orientation.
		y_hat = y0 + ( t / (T) ) * (y_star-y0)
		ys.append(y_hat)

	return ys

def ik_traj(q0, y_star, T, sim, viewer, ee_index, ndof=6, testing=False):
	"""
		TODO: fix.
		trajectory is proper, though PD isn't working.
	"""
	original_sim_state = sim.get_state()

	y0 = np.array(copy.deepcopy(sim.data.body_xpos[ee_index]))
	y0_quat = np.array(copy.deepcopy(sim.data.body_xquat[ee_index]))
	y_star = np.array(y_star)
	ys=[]
	for t in tqdm(np.arange(1,T)):

		y = copy.deepcopy(sim.data.body_xpos[ee_index])

		# # % compute Jacobian
		jacp,jacr=jac(sim, ee_index, ndof)

		# % new ee pose
		y_hat = y0 + ( t / (T) ) * (y_star-y0)

		# % new joint positions
		# TODO: orn
		J_pos=np.array(jacp)
		q_update = pseudoinv(J_pos).dot( (y_hat - y).reshape(3,1) )
		q = q + q_update[:ndof].reshape(ndof,)
		qs.append(q)

		full_q = np.concatenate((q, np.zeros(6)))
		full_qdot = np.zeros(12)

		torques = pd(None, full_qdot, full_q, sim, ndof=len(sim.data.ctrl))
		sim.data.ctrl[:ndof]=torques[:ndof]
		sim.step()
		viewer.render()

	qs=np.array(qs)

	# %% reset the simulation
	sim.set_state(original_sim_state)
	return qs

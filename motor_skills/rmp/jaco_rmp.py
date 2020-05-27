import numpy as np
import PyKDL as kdl
from motor_skills.rmp.rmp import RMPRoot
from motor_skills.rmp.rmp import RMPNode
from motor_skills.rmp.rmp_leaf import GoalAttractorUni
from motor_skills.rmp.rmp_leaf import CollisionAvoidance
from urdf_parser_py.urdf import URDF
from kdl_parser_py import urdf as parser
import abc

path = 'assets/kinova_j2s6s300/ros-j2s6s300.xml'


class JacoRMP(abc.ABC):
    def __init__(self):
        # parse out KDL kinematic chain from URDF
        robot = URDF.from_xml_file(path)
        base_link = "world"
        # end_link = "j2s6s300_link_6"
        end_link = "j2s6s300_link_finger_tip_1"
        # tree = kdl_tree_from_urdf_model(robot)
        _, tree = parser.treeFromUrdfModel(robot)
        self.chain = tree.getChain(base_link, end_link)

    @abc.abstractmethod
    def eval(self, q, qd):
        pass


class JacoFlatRMP(JacoRMP):
    def __init__(self):
        super().__init__()

        # import kinematics solvers
        self.pos_solver = kdl.ChainFkSolverPos_recursive(self.chain)
        self.jac_solver = kdl.ChainJntToJacSolver(self.chain)
        self.jacd_solver = kdl.ChainJntToJacDotSolver(self.chain)

        self.root = RMPRoot("jaco_root")

        # forward kinematics
        def psi(q):
            p_frame = kdl.Frame()
            jnt_q = np_to_jnt_arr(q)
            self.pos_solver.JntToCart(jnt_q, p_frame)
            p = p_frame.p
            # rz, ry, rx = p_frame.M.GetEulerZYX()
            # return np.array([[p.x(), p.y(), p.z(), rx, ry, rz]]).T
            return np.array([[p.x(), p.y(), p.z()]]).T

        # Jacobian for forward kinematics
        def J(q):
            # set of solver inputs
            nq = np.size(q)
            jnt_q = np_to_jnt_arr(q)
            jac = kdl.Jacobian(nq)

            # solve Jacobian and transfer into np array
            self.jac_solver.JntToJac(jnt_q, jac)
            return jac_to_np(jac)

        # Jacobian time-derivative of forward kinematics
        def J_dot(q, qd):
            # set solver inputs
            nq = np.size(q)
            jnt_q = np_to_jnt_arr(q)
            jnt_qd = np_to_jnt_arr(qd)
            jnt_q_qd = kdl.JntArrayVel(jnt_q, jnt_qd)
            jacd = kdl.Jacobian(nq)

            # solve and convert to np array
            self.jacd_solver.JntToJacDot(jnt_q_qd, jacd)
            return jac_to_np(jacd)

        self.hand = RMPNode("hand", self.root, psi, J, J_dot)
        self.atrc = GoalAttractorUni("jaco_attractor", self.hand, np.array([0]*3).T, gain=4)
        self.obst = CollisionAvoidance("jaco_avoider", self.hand, None, np.array([0]*3), R=0.05, alpha=2, eta=100)

    def eval(self, q, qd):
        # turn list inputs into column vectors and set state
        np_qdd = self.root.solve(np.array([q]).T, np.array([qd]).T)
        return np_qdd.flatten().tolist()

    def set_goal(self, goal):
        self.atrc.update_goal(np.array([goal]).T)

    def set_obst(self, obst):
        self.obst.set_obstacle(np.array([obst]).T)

class JacoTreeRMP(JacoRMP):
    def __init__(self):
        # TODO: set initial state
        # initialize solver for forward kinematic calculations/Jacobian calculations

        # set up main branch of tree to mimic kinematic chain
        root = RMPRoot("jaco_root")
        link1 = RMPNode("jaco_link1", root, None, None, None)
        root.add_child(link1)

        link2 = RMPNode("jaco_link2", link1, None, None, None)
        link1.add_child(link2)

        link3 = RMPNode("jaco_link3", link2, None, None, None)
        link2.add_child(link3)

        link4 = RMPNode("jaco_link4", link3, None, None, None)
        link3.add_child(link4)

        link5 = RMPNode("jaco_link5", link4, None, None, None)
        link4.add_child(link5)

        link6 = RMPNode("jaco_link6", link5, None, None, None)
        link5.add_child(link6)

    # evaluate jaco rmp for next joint control
    def eval(self, q, qd):
        return [0] * 6


def np_to_jnt_arr(arr):
    nq = np.size(arr)
    jnt_arr = kdl.JntArray(nq)
    for i in range(0, nq):
        jnt_arr[i] = arr[i]

    return jnt_arr

def jac_to_np(jac):
    nq = jac.columns()
    # used to be 6 to include rotation`
    np_jac = np.zeros((3, nq))
    for c in range(0, nq):
        c_twst = jac.getColumn(c)
        # used to be 6 to include rotation
        for r in range(0, 3):
            np_jac[r][c] = c_twst[r]

    return np_jac

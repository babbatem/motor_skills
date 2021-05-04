import time
import numpy as np

import pybullet as p
import pybullet_data

#from ompl import util as ompl_util
from ompl import base as ompl_base
from ompl import geometric as ompl_geo

NDOF = 6
FDOF = 6
URDFPATH='/home/mcorsaro/.mujoco/motor_skills/motor_skills/planner/assets/kinova_j2s6s300/j2s6s300.urdf'

def pbsetup():
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-10)
    p.loadURDF(URDFPATH, useFixedBase=True)
    p.loadURDF("plane.urdf", [0, 0, 0])
    return

# class for collision checking in python
# not yet doing multiple physics clients (though, should be fine for multiple instances of python?)
# ASSUMES robotId is 0, planeId is 1.
# ASSUMES NDOF DoF
class pbValidityChecker(ompl_base.StateValidityChecker):
    def __init__(self, si):
        super(pbValidityChecker, self).__init__(si)

        self.lower = np.array([p.getJointInfo(0, i)[8] for i in range(NDOF)])
        self.upper = np.array([p.getJointInfo(0, i)[9] for i in range(NDOF)])

        self.open_finger_state = [0., 0., 0., 0., 0., 0.]

    def resetFingerState(self):
        for i in range(FDOF):
            # reset joints with 0 velocity
            p.resetJointState(0, i+NDOF, self.open_finger_state[i], 0)

    def resetRobot(self, state, velocity=None):
        # TODO(mcorsaro): make this compatible with all "state" types (ompl.base._base.RealVectorStateInternal)
        '''if len(state) != NDOF:
            print("Resetting robot with states of size", len(state), "but robot is of size", NDOF)
            sys.exit()'''
        for i in range(NDOF):
            if velocity is not None:
                p.resetJointState(0,i,state[i],velocity)
            else:
                p.resetJointState(0,i,state[i])
        self.resetFingerState()

    # sets state and checks joint limits and collision
    def isValid(self, state):

        self.resetRobot(state, velocity=0)

        p.stepSimulation()
        return (
                self.check_plane_collision() and \
                self.check_joint_limits(state)
               )

    # Returns True if there is no collision with plane
    def check_plane_collision(self):
        contactPoints = p.getContactPoints(0, 1)
        if len(contactPoints) > 0:
            return False
        else:
            return True

    def check_joint_limits(self, state):
        for i in range(NDOF):
            if state[i] > self.upper[i] or state[i] < self.lower[i]:
                return False
        return True

    # Returns a valid state
    def sample_state(self):
        q = np.random.random(NDOF)*(self.upper - self.lower) + self.lower
        if self.isValid(q):
            return q
        else:
            return self.sample_state()

class PRMGenerator(object):
    """
        constructs pybullet simulation & plans therein with OMPL.
    """
    def __init__(self, prm_filename=None):
        super(PRMGenerator, self).__init__()

        self.ik_thresh = 0.0001
        self.ik_max_iter = 10000
        self.ik_rot_thresh = 0.0001

        # setup pybullet
        #p.connect(p.GUI)
        p.connect(p.DIRECT)
        pbsetup()

        # setup space
        lower = np.array([p.getJointInfo(0, i)[8] for i in range(NDOF)])
        upper = np.array([p.getJointInfo(0, i)[9] for i in range(NDOF)])
        bounds = ompl_base.RealVectorBounds(NDOF)
        for i in range(NDOF):
            bounds.setLow(i,lower[i])
            bounds.setHigh(i,upper[i])

        self.space = ompl_base.RealVectorStateSpace(NDOF)
        self.space.setBounds(bounds)

        # Construct a space information instance for this state space
        self.si = ompl_base.SpaceInformation(self.space)

        self.validityChecker = pbValidityChecker(self.si)
        self.si.setStateValidityChecker(self.validityChecker)
        self.si.setup()

        if prm_filename is not None:
            storage = ompl_base.PlannerDataStorage()
            planner_data = ompl_base.PlannerData(self.si)
            storage.load(prm_filename, planner_data)
            self.optimizingPlanner = ompl_geo.LazyPRMstar(planner_data)
        else:
            self.optimizingPlanner = ompl_geo.LazyPRMstar(self.si)

        if prm_filename is None:
            # no filename specified, not loading PRM, so assume saving and generate a bunch of states
            self.runTime = 60.0*30
        else:
            # filename specified, load PRM, plan fast
            self.runTime = 0.01

        self.setup = False

    def accurateCalculateInverseKinematics(self, robotId, endEffectorIndex, targetPos, targetQuat, starting_state=None):
        if starting_state is not None:
            self.validityChecker.resetRobot(starting_state, velocity=0)
        closeEnough = False
        c_iter = 0
        dist2 = 1e30
        while (not closeEnough and c_iter < self.ik_max_iter):
            jointPoses = p.calculateInverseKinematics(robotId, endEffectorIndex, targetPos, targetQuat)
            self.validityChecker.resetRobot(jointPoses)
            new_pos, new_quat = self.calculateForwardKinematics(robotId, endEffectorIndex)
            diff = [targetPos[0] - new_pos[0], targetPos[1] - new_pos[1], targetPos[2] - new_pos[2]]
            dist2 = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
            np_nq, np_tq = np.array(new_quat), np.array(targetQuat)
            rot_diff = 2*np.arccos(np.dot(np_nq, np_tq))
            closeEnough = (dist2 < self.ik_thresh) and (rot_diff < self.ik_rot_thresh)
            c_iter = c_iter + 1
        #print ("Num iter:", c_iter, "threshold:", dist2, rot_diff)
        return jointPoses

    def calculateForwardKinematics(self, robotId, endEffectorIndex, joint_state=None):
        if joint_state is not None:
            self.validityChecker.resetRobot(joint_state, velocity=0)
        link_state = p.getLinkState(robotId, endEffectorIndex)
        ee_pos, ee_quat = link_state[4:6]
        return (ee_pos, ee_quat)

    def plan(self, start_q, goal_q, check_validity=True):

        if check_validity:
            if (not self.validityChecker.isValid(start_q)):
                raise Exception("Start joints put robot in collision.")
            if (not self.validityChecker.isValid(goal_q)):
                raise Exception("Goal joints put robot in collision.")

        self.validityChecker.resetRobot(start_q, velocity=0)

        # start and goal configs
        start = ompl_base.State(self.space)
        for i in range(len(start_q)):
            start[i] = start_q[i]

        goal = ompl_base.State(self.space)
        for i in range(len(start_q)):
            goal[i] = goal_q[i]

        # setup and solve
        pdef = ompl_base.ProblemDefinition(self.si)
        pdef.setOptimizationObjective(ompl_base.PathLengthOptimizationObjective(self.si))
        pdef.setStartAndGoalStates(start, goal)

        self.optimizingPlanner.setProblemDefinition(pdef)
        if not self.setup:
            self.optimizingPlanner.setup()
            self.setup = True
        solved = self.optimizingPlanner.solve(self.runTime)

        solution = None
        if solved:
            # Output the length of the path found
            print('{0} found solution of path length {1:.4f} with an optimization ' \
                'objective value of {2:.4f}'.format( \
                self.optimizingPlanner.getName(), \
                pdef.getSolutionPath().length(), \
                pdef.getSolutionPath().cost(pdef.getOptimizationObjective()).value()))

            solution = pdef.getSolutionPath()

        else:
            print("No solution found.")
        self.optimizingPlanner.clearQuery()
        return solution

    def savePRM(self, filename="/home/mcorsaro/Desktop/TESTPRM"):
        planner_data = ompl_base.PlannerData(self.optimizingPlanner.getSpaceInformation())
        self.optimizingPlanner.getPlannerData(planner_data)
        storage = ompl_base.PlannerDataStorage()
        storage.store(planner_data, filename)

def generatePRM():
    #np.random.seed(0)
    save = False
    prmfile= None if save else "/home/mcorsaro/Desktop/TESTPRM"
    prm_gen = PRMGenerator(prmfile)
    for i in range(1):
        s = prm_gen.validityChecker.sample_state()
        g = prm_gen.validityChecker.sample_state()
        result = prm_gen.plan(s, g)
    if save:
        prm_gen.savePRM()

if __name__ == '__main__':
    generatePRM()

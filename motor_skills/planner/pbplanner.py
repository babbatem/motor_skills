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
DOORPATH='/home/mcorsaro/.mujoco/motor_skills/motor_skills/planner/assets/_frame.urdf'

def pbsetup():
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-10)
    p.loadURDF(URDFPATH, useFixedBase=True)
    p.loadURDF("plane.urdf", [0, 0, 0]) # This is also in URDF

    p.loadURDF(DOORPATH, [0.0, 0.5, 0.44], useFixedBase=True) # note: this is hardcoded, pulled from URDF
    return

# class for collision checking in python
# not yet doing multiple physics clients (though, should be fine for multiple instances of python?)
# ASSUMES robotId is 0, planeId is 1.
# ASSUMES NDOF DoF
class pbValidityChecker(ompl_base.StateValidityChecker):
    def __init__(self, si, door_uid=None):
        super(pbValidityChecker, self).__init__(si)

        self.lower = np.array([p.getJointInfo(0, i)[8] for i in range(NDOF)])
        self.upper = np.array([p.getJointInfo(0, i)[9] for i in range(NDOF)])

        self.otherIds = []
        self.otherObj_states = {} # maps Uid to reset states
        self.otherObj_dofs = {} # maps Uid to joint indices to be reset

        self.checking_other_ids = True
        if door_uid is not None:
            self.otherIds.append(door_uid)
            self.otherObj_states[door_uid] = [0,0]
            self.otherObj_dofs[door_uid] = [0,2]

        self.open_finger_state = None
        self.current_finger_state = None

    def resetFingerState(self):
        for i in range(FDOF):
            # reset joints with 0 velocity
            p.resetJointState(0, i+NDOF, self.current_finger_state[i], 0)

    def updateFingerState(self, finger_state):
        self.current_finger_state = finger_state
        self.resetFingerState()

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

    def resetScene(self):
        # TODO: more general
        for i in range(len(self.otherIds)):
            for j in range(len(self.otherObj_dofs[self.otherIds[i]])):
                p.resetJointState(self.otherIds[i],
                                  self.otherObj_dofs[self.otherIds[i]][j],
                                  self.otherObj_states[self.otherIds[i]][j])

    # sets state and checks joint limits and collision
    def isValid(self, state):

        self.resetRobot(state, velocity=0)
        self.resetScene()

        p.stepSimulation()
        other_ids_to_check = self.otherIds if self.checking_other_ids else []
        return (
                self.detect_collisions(other_ids_to_check) and \
                self.check_plane_collision() and \
                self.check_joint_limits(state)
               )

    # sets state and checks joint limits and collision, and returns code as to why
    def isInvalid(self, state):

        self.resetRobot(state, velocity=0)
        self.resetScene()

        p.stepSimulation()
        # if True, no collisions
        if not self.detect_collisions(self.otherIds):
            return 1
        if not self.check_plane_collision():
            return 2
        if not self.check_joint_limits(state):
            return 3
        return 0

    # Returns True if there is no collision with plane
    def check_plane_collision(self):
        contactPoints = p.getContactPoints(0, 1)
        if len(contactPoints) > 0:
            return False
        else:
            return True

    # Recursively checks for collision between robotId and ids_to_check
    def detect_collisions(self, ids_to_check):

        if len(ids_to_check) == 0:
            return True

        else:
            contactPoints = p.getContactPoints(0, ids_to_check[-1])

            if len(contactPoints) > 0:
                return False
            else:
                return self.detect_collisions(ids_to_check[0:-1])

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

class PbPlanner(object):
    """
        constructs pybullet simulation & plans therein with OMPL.
    """
    def __init__(self, obj, prm_filename):
        super(PbPlanner, self).__init__()
        self.obj = obj

        if self.obj != 'door':
            print("Objects other than door not implemented in pybullet")
            return
        self.ik_thresh = 0.0001
        self.ik_max_iter = 10000
        self.ik_rot_thresh = 0.00001

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

        # Set the object used to check which states in the space are valid
        # TODO: Lookup door Uid. Currently assume it's 2 because robot is 0 and plane is 1
        if self.obj == 'door':
            door_uid = 2
            self.validityChecker = pbValidityChecker(self.si, door_uid)
            self.si.setStateValidityChecker(self.validityChecker)
            self.si.setup()

        storage = ompl_base.PlannerDataStorage()
        planner_data = ompl_base.PlannerData(self.si)
        storage.load(prm_filename, planner_data)
        self.optimizingPlanner = ompl_geo.LazyPRMstar(planner_data)

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
            dist2, rot_diff = self.distBetweenPoses(targetPos, new_pos, targetQuat, new_quat)
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

    def distBetweenPoses(self, pos1, pos2, quat1, quat2):
        diff = [pos1[0] - pos2[0], pos1[1] - pos2[1], pos1[2] - pos2[2]]
        dist2 = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
        np_quat1, np_quat2 = np.array(quat1), np.array(quat2)
        rot_diff = np.abs(2*np.arccos(np.dot(np_quat1, np_quat2)))
        return dist2, rot_diff

    def plan(self, start_q, goal_q, check_validity=True):

        if check_validity:
            if (not self.validityChecker.isValid(start_q)):
                raise Exception("Start joints put robot in collision.")
            if (not self.validityChecker.isValid(goal_q)):
                raise Exception("Goal joints put robot in collision.")

        self.validityChecker.resetRobot(start_q, velocity=0)
        self.validityChecker.resetScene()

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

def demo():

    planner = PbPlanner()
    s = planner.validityChecker.sample_state()
    g = planner.validityChecker.sample_state()

    # plan
    result=planner.plan(s, g)
    p.disconnect()

    # visualize plan
    p.connect(p.GUI)
    pbsetup()

    for i in range(len(s)):
        p.resetJointState(0,i,s[i],0)
    p.stepSimulation()
    _=input('Start state. Press enter to continue')

    for i in range(len(g)):
        p.resetJointState(0,i,g[i],0)
    p.stepSimulation()
    _=input('Goal state. Press enter to continue')

    for i in range(len(s)):
        p.resetJointState(0,i,s[i],0)
    p.stepSimulation()

    result.interpolate(1000)
    H = result.getStateCount()
    print(H)
    for t in range(H):
        state_t = result.getState(t)
        for i in range(NDOF):
            p.resetJointState(0, i, state_t[i],0)
        p.stepSimulation()
        time.sleep(0.01)


    _=input('Press enter to visualize goal again ')
    for i in range(len(g)):
        p.resetJointState(0,i,g[i],0)
    p.stepSimulation()

    _=input('Press enter to exit ')
    p.disconnect()

def execution_test():
    planner = PbPlanner()
    s = planner.validityChecker.sample_state()
    g = planner.validityChecker.sample_state()

    # plan
    result=planner.plan(s, g)
    p.disconnect()

    # visualize plan
    p.connect(p.GUI)
    pbsetup()

    for i in range(len(s)):
        p.resetJointState(0,i,s[i],0)
    p.stepSimulation()
    _=input('Start state. Press enter to continue')

    for i in range(len(g)):
        p.resetJointState(0,i,g[i],0)
    p.stepSimulation()
    _=input('Goal state. Press enter to continue')

    for i in range(len(s)):
        p.resetJointState(0,i,s[i],0)
    p.stepSimulation()

    result.interpolate(100)
    H = result.getStateCount()
    print(H)
    for t in range(H):
        state_t = result.getState(t)
        for i in range(NDOF):
            p.setJointMotorControl2(0,i,p.POSITION_CONTROL,targetPosition=state_t[i], positionGain=1)
        p.stepSimulation()
        time.sleep(0.01)

    _=input('Press enter to visualize goal again ')
    for i in range(len(g)):
        p.resetJointState(0,i,g[i],0)
    p.stepSimulation()

    _=input('Press enter to exit ')
    p.disconnect()


if __name__ == '__main__':
    # demo()
    execution_test()

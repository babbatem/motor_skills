import time
import numpy as np

import pybullet as p
import pybullet_data

#from ompl import util as ompl_util
from ompl import base as ompl_base
#from ompl import geometric as ompl_geo

from motor_skills.planner.ompl_optimal_demo import allocateObjective, allocatePlanner, getPathLengthObjWithCostToGo, getPathLengthObjective

NDOF = 6
URDFPATH='/home/mcorsaro/.mujoco/motor_skills/motor_skills/planner/assets/kinova_j2s6s300/j2s6s300.urdf'
DOORPATH='/home/mcorsaro/.mujoco/motor_skills/motor_skills/planner/assets/_frame.urdf'

def pbsetup(load_door):
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-10)
    p.loadURDF(URDFPATH, useFixedBase=True)
    p.loadURDF("plane.urdf", [0, 0, 0])
    if load_door:
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
        if door_uid:
            self.otherIds.append(door_uid)
            self.otherObj_states[door_uid] = [0,0]
            self.otherObj_dofs[door_uid] = [0,2]

    def resetRobot(self, state):
        # TODO(mcorsaro): make this compatible with all "state" types (ompl.base._base.RealVectorStateInternal)
        '''if len(state) != NDOF:
            print("Resetting robot with states of size", len(state), "but robot is of size", NDOF)
            sys.exit()'''
        for i in range(NDOF):
            p.resetJointState(0,i,state[i],0)

    def resetScene(self):
        # TODO: more general
        for i in range(len(self.otherIds)):
            for j in range(len(self.otherObj_dofs[self.otherIds[i]])):
                p.resetJointState(self.otherIds[i],
                                  self.otherObj_dofs[self.otherIds[i]][j],
                                  self.otherObj_states[self.otherIds[i]][j])

    # sets state and checks joint limits and collision
    def isValid(self, state):

        self.resetRobot(state)
        self.resetScene()

        p.stepSimulation()
        return (
                self.detect_collisions(self.otherIds) and \
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
    def __init__(self, load_door=False):
        super(PbPlanner, self).__init__()

        self.ik_thresh = 0.0001
        self.ik_max_iter = 10000

        # setup pybullet
        # p.connect(p.GUI)
        p.connect(p.DIRECT)
        pbsetup(load_door)

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
        door_uid = 2 if load_door else None
        self.validityChecker = pbValidityChecker(self.si, door_uid)
        self.si.setStateValidityChecker(self.validityChecker)
        self.si.setup()

        self.runTime = 5.0
        self.plannerType = 'RRTstar'

    def accurateCalculateInverseKinematics(self, robotId, endEffectorIndex, targetPos, targetQuat):
        closeEnough = False
        c_iter = 0
        dist2 = 1e30
        while (not closeEnough and c_iter < self.ik_max_iter):
            jointPoses = p.calculateInverseKinematics(robotId, endEffectorIndex, targetPos, targetQuat)
            for i in range(NDOF):
                p.resetJointState(robotId, i, jointPoses[i])
            ls = p.getLinkState(robotId, endEffectorIndex)
            newPos = ls[4]
            diff = [targetPos[0] - newPos[0], targetPos[1] - newPos[1], targetPos[2] - newPos[2]]
            dist2 = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
            closeEnough = (dist2 < self.ik_thresh)
            c_iter = c_iter + 1
        #print ("Num iter: "+str(iter) + "threshold: "+str(dist2))
        return jointPoses

    # goal ee xyz position
    # goal ee wxyz orientation
    def get_ik_pose(self, goal_position, goal_orientation, verbose=False):

        #get accurate solution not including orientation
        s = self.accurateCalculateInverseKinematics(0, NDOF, goal_position, goal_orientation)
        #set joints
        for i in range(len(s)):
            p.resetJointState(0,i,s[i],0)
        p.stepSimulation()

        return(s)

    def plan(self, start_q, goal_q):

        if (not self.validityChecker.isValid(start_q)):
            raise Exception("Start joints put robot in collision.")
        if (not self.validityChecker.isValid(goal_q)):
            raise Exception("Goal joints put robot in collision.")

        self.validityChecker.resetRobot(start_q)
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
        pdef.setStartAndGoalStates(start, goal)
        pdef.setOptimizationObjective(getPathLengthObjective(self.si))
        optimizingPlanner = allocatePlanner(self.si, self.plannerType)
        optimizingPlanner.setProblemDefinition(pdef)
        optimizingPlanner.setup()
        solved = optimizingPlanner.solve(self.runTime)

        if solved:
            # Output the length of the path found
            print('{0} found solution of path length {1:.4f} with an optimization ' \
                'objective value of {2:.4f}'.format( \
                optimizingPlanner.getName(), \
                pdef.getSolutionPath().length(), \
                pdef.getSolutionPath().cost(pdef.getOptimizationObjective()).value()))

            return pdef.getSolutionPath()

        else:
            print("No solution found.")
            return None

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

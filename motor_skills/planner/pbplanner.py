import time
import numpy as np

import pybullet as p
import pybullet_data

from ompl import util as ou
from ompl import base as ob
from ompl import geometric as og

from motor_skills.planner.ompl_optimal_demo import allocateObjective, allocatePlanner, getPathLengthObjWithCostToGo, getPathLengthObjective

NDOF = 6
URDFPATH='/home/abba/msu_ws/src/motor_skills/motor_skills/planner/assets/kinova_j2s6s300/j2s6s300.urdf'
DOORPATH='/home/abba/msu_ws/src/motor_skills/motor_skills/planner/assets/_frame.urdf'

def pbsetup():
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-10)
    p.loadURDF(URDFPATH, useFixedBase=True)
    p.loadURDF("plane.urdf", [0, 0, 0])
    p.loadURDF(DOORPATH, [0.0, 0.5, 0.44], useFixedBase=True) # note: this is hardcoded, pulled from URDF
    return

# class for collision checking in python
# not yet doing multiple physics clients (though, should be fine for multiple instances of python?)
# ASSUMES robotId is 0, planeId is 1.
# ASSUMES NDOF DoF
class pbValidityChecker(ob.StateValidityChecker):
    def __init__(self, si, otherIds):
        super(pbValidityChecker, self).__init__(si)
        self.otherIds = otherIds
        self.lower = np.array([p.getJointInfo(0, i)[8] for i in range(NDOF)])
        self.upper = np.array([p.getJointInfo(0, i)[9] for i in range(NDOF)])

        self.otherObj_states = { 2: [0,0] } # maps Uid to reset states
        self.otherObj_dofs =   { 2: [0,2] } # maps Uid to joint indices to be reset

    # sets state and checks joint limits and collision
    def isValid(self, state):

        # first reset robot state
        for i in range(NDOF):
            p.resetJointState(0,i,state[i],0)

        # then reset scene state.
        # TODO: more general
        for i in range(len(self.otherIds)):
            for j in range(len(self.otherObj_dofs[self.otherIds[i]])):
                p.resetJointState(self.otherIds[i],
                                  self.otherObj_dofs[self.otherIds[i]][j],
                                  self.otherObj_states[self.otherIds[i]][j])


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
    def __init__(self):
        super(PbPlanner, self).__init__()

        # setup pybullet
        # p.connect(p.GUI)
        p.connect(p.DIRECT)
        pbsetup()

        # setup space
        lower = np.array([p.getJointInfo(0, i)[8] for i in range(NDOF)])
        upper = np.array([p.getJointInfo(0, i)[9] for i in range(NDOF)])
        bounds = ob.RealVectorBounds(NDOF)
        for i in range(NDOF):
            bounds.setLow(i,lower[i])
            bounds.setHigh(i,upper[i])

        self.space = ob.RealVectorStateSpace(NDOF)
        self.space.setBounds(bounds)

        # Construct a space information instance for this state space
        self.si = ob.SpaceInformation(self.space)

        # Set the object used to check which states in the space are valid
        self.validityChecker = pbValidityChecker(self.si, [2])
        self.si.setStateValidityChecker(self.validityChecker)
        self.si.setup()

        self.runTime = 5.0
        self.plannerType = 'RRTstar'

    def plan(self, start_q, goal_q):

        # start and goal configs
        start = ob.State(self.space)
        for i in range(len(start_q)):
            start[i] = start_q[i]

        goal = ob.State(self.space)
        for i in range(len(start_q)):
            goal[i] = goal_q[i]

        # setup and solve
        pdef = ob.ProblemDefinition(self.si)
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

import time
import numpy as np

import pybullet as p
import pybullet_data

from ompl import util as ou
from ompl import base as ob
from ompl import geometric as og

from ompl_optimal_demo import allocateObjective, allocatePlanner, getPathLengthObjWithCostToGo, getPathLengthObjective

NDOF = 6
URDFPATH='/home/abba/msu_ws/src/motor_skills/motor_skills/planner/assets/kinova_j2s6s300/j2s6s300.urdf'

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

    # sets state and checks joint limits and collision
    def isValid(self, state):
        for i in range(NDOF):
            p.resetJointState(0,i,state[i],0)
        p.stepSimulation()
        return (
                self.detect_collisions(self.otherIds) and \
                self.check_plane_collision() and \
                self.check_joint_limits(state)
               )


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
                return self.isValid(ids[0:-1])

    def check_joint_limits(self, state):
        for i in range(NDOF):
            if state[i] > self.upper[i] or state[i] < self.lower[i]:
                return False
        return True

    def sample_state(self):

        q = np.random.random(NDOF)*(self.upper - self.lower) + self.lower
        if self.isValid(q):
            return q
        else:
            return self.sample_state()

# TODO
class PbPlanner(object):
    """docstring for PbPlanner."""
    def __init__(self, arg):
        super(PbPlanner, self).__init__()
        self.arg = arg


def plan(runTime, plannerType, start_q=np.zeros(NDOF), goal_q=np.ones(NDOF)):

    lower = np.array([p.getJointInfo(0, i)[8] for i in range(NDOF)])
    upper = np.array([p.getJointInfo(0, i)[9] for i in range(NDOF)])
    bounds = ob.RealVectorBounds(NDOF)
    for i in range(NDOF):
        bounds.setLow(i,lower[i])
        bounds.setHigh(i,upper[i])

    space = ob.RealVectorStateSpace(NDOF)
    space.setBounds(bounds)

    # Construct a space information instance for this state space
    si = ob.SpaceInformation(space)

    # Set the object used to check which states in the space are valid
    validityChecker = pbValidityChecker(si, [])
    si.setStateValidityChecker(validityChecker)

    si.setup()

    # assume start and goal configs
    start = ob.State(space)
    for i in range(len(start_q)):
        start[i] = start_q[i]

    goal = ob.State(space)
    for i in range(len(start_q)):
        goal[i] = goal_q[i]

    # Create a problem instance
    pdef = ob.ProblemDefinition(si)

    # Set the start and goal states
    pdef.setStartAndGoalStates(start, goal)

    # Create the optimization objective specified by our command-line argument.
    # This helper function is simply a switch statement.
    # pdef.setOptimizationObjective(getPathLengthObjWithCostToGo(si))
    pdef.setOptimizationObjective(getPathLengthObjective(si))

    # Construct the optimal planner specified by our command line argument.
    # This helper function is simply a switch statement.
    optimizingPlanner = allocatePlanner(si, plannerType)

    # Set the problem instance for our planner to solve
    optimizingPlanner.setProblemDefinition(pdef)
    optimizingPlanner.setup()

    # attempt to solve the planning problem in the given runtime
    solved = optimizingPlanner.solve(runTime)

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


def pbsetup():
    # p.connect(p.GUI)
    # p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-10)

    robotId = p.loadURDF(URDFPATH, useFixedBase=True)
    planeId = p.loadURDF("plane.urdf")

if __name__ == '__main__':
    p.connect(p.DIRECT)
    pbsetup()

    lower = np.array([p.getJointInfo(0, i)[8] for i in range(NDOF)])
    upper = np.array([p.getJointInfo(0, i)[9] for i in range(NDOF)])
    bounds = ob.RealVectorBounds(NDOF)
    for i in range(NDOF):
        bounds.setLow(i,lower[i])
        bounds.setHigh(i,upper[i])

    space = ob.RealVectorStateSpace(NDOF)
    space.setBounds(bounds)
    si = ob.SpaceInformation(space)

    validityChecker = pbValidityChecker(si, [])

    s = validityChecker.sample_state()
    g = validityChecker.sample_state()

    result=plan(1.0, 'RRTstar',
                start_q = s, goal_q = g)

    p.disconnect()

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
            p.resetJointState(0, i, state_t[i],0)
        p.stepSimulation()
        time.sleep(0.01)


    _=input('Press enter to visualize goal again ')
    for i in range(len(g)):
        p.resetJointState(0,i,g[i],0)
    p.stepSimulation()

    _=input('Press enter to exit ')
    p.disconnect()

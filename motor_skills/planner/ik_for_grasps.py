import time
import pickle
import numpy as np
import random
import pickle

import pybullet as p
import pybullet_data

'''
class DummyPbPlanner(object):
    """
        constructs pybullet simulation & plans therein with OMPL.
    """
    def __init__(self):
        super(DummyPbPlanner, self).__init__()

        # setup pybullet
        # p.connect(p.GUI)
        p.connect(p.DIRECT)
        pbsetup()
'''

class PBIK(object):
    def __init__(self, ndof, urdf_path, door_path, thresh, max_iter):
        self.ndof = ndof
        self.urdf_path = urdf_path
        self.door_path = door_path

        self.thresh = thresh
        self.max_iter = max_iter

    def pbsetup(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-10)
        robot_idx = p.loadURDF(self.urdf_path, useFixedBase=True)
        p.loadURDF("plane.urdf", [0, 0, 0])
        p.loadURDF(self.door_path, [0.0, 0.5, 0.44], useFixedBase=True) # note: this is hardcoded, pulled from URDF
        return

    def accurateCalculateInverseKinematics(self, kukaId, kukaEndEffectorIndex, targetPos, targetQuat):
        closeEnough = False
        c_iter = 0
        dist2 = 1e30
        while (not closeEnough and c_iter < self.max_iter):
            jointPoses = p.calculateInverseKinematics(kukaId, kukaEndEffectorIndex, targetPos, targetQuat)
            for i in range(self.ndof):
                p.resetJointState(kukaId, i, jointPoses[i])
            ls = p.getLinkState(kukaId, kukaEndEffectorIndex)
            newPos = ls[4]
            diff = [targetPos[0] - newPos[0], targetPos[1] - newPos[1], targetPos[2] - newPos[2]]
            dist2 = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
            closeEnough = (dist2 < self.thresh)
            c_iter = c_iter + 1
        #print ("Num iter: "+str(iter) + "threshold: "+str(dist2))
        return jointPoses

    # goal ee xyz position
    # goal ee wxyz orientation
    def get_ik_pose(self, goal_position, goal_orientation):

        # visualize plan
        p.connect(p.GUI)
        #p.connect(p.DIRECT)
        self.pbsetup()


        #get accurate solution not including orientation
        s = self.accurateCalculateInverseKinematics(0, self.ndof, goal_position, goal_orientation)
        #set joints
        for i in range(len(s)):
            #p.resetJointState(0,i,s[i],0)
            p.resetJointState(0,i,s[i],0)
        p.stepSimulation()

        #Info on eepose after setting joints
        eepose = p.getLinkState(0,6)
        print("ee pose state: ",eepose)
        _=input('Press enter to exit ')
        p.disconnect()

        return(s)
    
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

def demo():
    # TODO(mcorsaro): numJoints = p.getNumJoints(kukaId)
    ndof = 6
    parent_path = "/home/mcorsaro/.mujoco/motor_skills/motor_skills/"
    urdf_path = parent_path + 'planner/assets/kinova_j2s6s300/j2s6s300.urdf'
    door_path = parent_path + 'planner/assets/_frame.urdf'
    thresh = 0.0001
    max_iter = 10000
    ik = PBIK(ndof, urdf_path, door_path, thresh, max_iter)
    ik.get_ik_pose([0, 0.2, 0.2], [1, 0, 0, 0])

    print("Done!")

if __name__ == '__main__':
    demo()

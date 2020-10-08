import time
import numpy as np

import pybullet as p
import pybullet_data


NDOF = 6
parent_path = "/home/eric/Github/motor_skills/motor_skills/"
URDFPATH= parent_path + 'planner/assets/kinova_j2s6s300/j2s6s300.urdf'
DOORPATH= parent_path + 'planner/assets/_frame.urdf'

def accurateCalculateInverseKinematics(kukaId, kukaEndEffectorIndex, targetPos, threshold, maxIter):
    closeEnough = False
    iter = 0
    dist2 = 1e30
    while (not closeEnough and iter < maxIter):
      jointPoses = p.calculateInverseKinematics(kukaId, kukaEndEffectorIndex, targetPos)
      for i in range(6):
        p.resetJointState(kukaId, i, jointPoses[i])
      ls = p.getLinkState(kukaId, kukaEndEffectorIndex)
      newPos = ls[4]
      diff = [targetPos[0] - newPos[0], targetPos[1] - newPos[1], targetPos[2] - newPos[2]]
      dist2 = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
      closeEnough = (dist2 < threshold)
      iter = iter + 1
    #print ("Num iter: "+str(iter) + "threshold: "+str(dist2))
    return jointPoses

def pbsetup():
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-10)
    robot_idx = p.loadURDF(URDFPATH, useFixedBase=True)
    p.loadURDF("plane.urdf", [0, 0, 0])
    p.loadURDF(DOORPATH, [0.0, 0.5, 0.44], useFixedBase=True) # note: this is hardcoded, pulled from URDF
    return

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

def execution_test():
    planner = PbPlanner()

    # plan
    p.disconnect()

    # visualize plan
    p.connect(p.GUI)
    pbsetup()

    #IK
    #supposedly good xyz and quat
    xyz = [0.31172854238601855, 0.3339689937782047, 0.43719009216390514]
    quat = [-0.6260310664250478, 0.7210224167750562, 0.22065623688446964, 0.1988029262927463]
    #xyz = [0.25886307824156046, 0.3426639263843749, 0.44332385048063455]
    #quat = [0.7684128458063817, -0.6285934875323844, 0.11482174898764229, -0.03504128694883987]


    #actually good s and associated xyz and quat
    #s = [2.6308434905854114, 3.2638783747295625, 1.6031654073824808, 2.589366055729374, 1.9375218288173555, 2.619735866509471]
    #xyz = [0.3107170168694746, 0.2772614148483482, 0.5196207402128867]
    #quat = [0.01565529478467831, -0.2964578815070268, 0.9525644916001882, 0.0669964594590313]
    s = accurateCalculateInverseKinematics(0,6,xyz,0.1,10000)
    for i in range(len(s)):
        #p.resetJointState(0,i,s[i],0)
        p.resetJointState(0,i,s[i],0)
    p.stepSimulation()
    s = p.calculateInverseKinematics(0,6,xyz,quat)
    for i in range(len(s)):
        #p.resetJointState(0,i,s[i],0)
        p.resetJointState(0,i,s[i],0)
    p.stepSimulation()
    #s = [2.0948452521356145, 3.4080710066614337, 1.9288332365499747, -2.9022845794290744, 1.5140629279333249, 2.311352661620965]



    #s = [-0.5878,  3.1631,  4.9078, -0.6735,  2.0874,  2.8526]


    eepose = p.getLinkState(0,6)
    print("ee pose state: ",eepose)
    _=input('Press enter to exit ')
    p.disconnect()


if __name__ == '__main__':
    execution_test()

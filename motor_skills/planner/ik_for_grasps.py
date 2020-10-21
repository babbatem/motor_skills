import time
import pickle
import numpy as np
import random
import pickle

import pybullet as p
import pybullet_data


NDOF = 6
parent_path = "/home/eric/Github/motor_skills/motor_skills/"
URDFPATH= parent_path + 'planner/assets/kinova_j2s6s300/j2s6s300.urdf'
DOORPATH= parent_path + 'planner/assets/_frame.urdf'


def accurateCalculateInverseKinematics(kukaId, kukaEndEffectorIndex, targetPos, targetQuat, threshold, maxIter):
    closeEnough = False
    iter = 0
    dist2 = 1e30
    while (not closeEnough and iter < maxIter):
      jointPoses = p.calculateInverseKinematics(kukaId, kukaEndEffectorIndex, targetPos, targetQuat)
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

def data_augmentation():
    #GPD_POSES_PATH = "/home/eric/Github/motor_skills/motor_skills/experiments/gpd_data_dict"
    GPD_POSES_PATH = "./good.p"
    #grasp_qs = pickle.load(open(GPD_POSES_PATH, "rb"))
    with open(GPD_POSES_PATH, 'rb') as f:
        grasp_qs = pickle.load(f, encoding="latin1")

    #num initial points
    num_grasps = len(grasp_qs['xyz'])

    #number of new points per existing point
    new = 500
    #radius of positions around cube
    r = 0.0
    #perturbation of quaternion
    q = 0.0

    new_data = {'xyz':[], 'quat':[], 'joint_pos':[]}
    for n in range(num_grasps):
    #    if n not in [4, 8, 12, 15, 24, 43, 70, 77, 83, 89, 118, 126, 131, 140, 141, 160, 184]:
    #        continue

        n_xyz = grasp_qs['xyz'][n]
        n_quat = grasp_qs['quat'][n]

        #Add original point to DA dict
        #new_data['xyz'].append(n_xyz)
        #new_data['quat'].append(n_quat)
        #new_data['joint_pos'].append(grasp_qs['joint_pos'][n])
        for _ in range(new):
            print("\n%s out of %s\n" % (n, num_grasps))
            x,y,z = n_xyz
            w,i,j,k = n_quat 
            
            #IK
            #desired ee pose
            '''
            xyz = [0.31172854238601855, 0.3339689937782047, 0.43719009216390514]
            quat = [-0.6260310664250478, 0.7210224167750562, 0.22065623688446964, 0.1988029262927463]

            x = xyz[0]
            y = xyz[1]
            z = xyz[2]
            w = quat[0]
            i = quat[1]
            j = quat[2]
            k = quat[3]
            '''

            x += random.uniform(-1*r,r)
            y += random.uniform(-1*r,r)
            z += random.uniform(-1*r,r)

            w += random.uniform(-1*q,q)
            i += random.uniform(-1*q,q)
            j += random.uniform(-1*q,q)
            k += random.uniform(-1*q,q)

            quat = list(normalize(np.array([w,i,j,k])))

            s = get_ik_pose([x,y,z],quat)

            new_data['xyz'].append(np.array([x,y,z]))
            new_data['quat'].append(np.array([w,i,j,k]))
            new_data['joint_pos'].append(np.array(s[:6]))

    return(new_data)


    
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def get_ik_pose(xyz,quat):
    planner = PbPlanner()
    # plan
    p.disconnect()

    # visualize plan
    p.connect(p.GUI)
    #p.connect(p.DIRECT)
    pbsetup()


    #get accurate solution not including orientation
    s = accurateCalculateInverseKinematics(0,6,xyz,quat,0.0001,10000)
    #set joints
    for i in range(len(s)):
        #p.resetJointState(0,i,s[i],0)
        p.resetJointState(0,i,s[i],0)
    p.stepSimulation()

    #Info on eepose after setting joints
    #eepose = p.getLinkState(0,6)
    #print("ee pose state: ",eepose)
    _=input('Press enter to exit ')
    p.disconnect()

    return(s)


if __name__ == '__main__':
    da_data = data_augmentation()
    pickle.dump(da_data,open("data_augmentation.p","wb"))

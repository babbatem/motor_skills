import os
import pathlib
import numpy as np
import time as timer
from tqdm import tqdm
import pickle

from trajopt.algos.mppi import MPPI
from trajopt.envs.utils import get_environment

import motor_skills
from motor_skills.envs.mj_jaco import MjJacoMPPI

class mppiPlanner(object):
    """docstring for mppiPlanner."""
    def __init__(self):
        super(mppiPlanner, self).__init__()
        self.start = None
        self.goal_pos = None
        self.goal_quat = None
        self.env_callable = MjJacoMPPI
        self.rollout_env_kwargs = None

    def plan(self, start, goal_pos, goal_quat):

        rollout_env_kwargs = {"start" : start,
                              "goal_pos": goal_pos,
                              "goal_quat" : goal_quat,
                              "vis" : False}

        vis_env_kwargs = {"start" : start,
                           "goal_pos": goal_pos,
                           "goal_quat" : goal_quat,
                           "vis" : True}

        # %% TODO: don't hardcode these paths
        parent_dir_path = str(pathlib.Path(__file__).parent.absolute())
        output = parent_dir_path + '/outputs/mppiPlanner'
        config = parent_dir_path + '/configs/jaco_mppi_config.txt'

        OUT_DIR = output
        if not os.path.exists(OUT_DIR):
            os.mkdir(OUT_DIR)
        with open(config, 'r') as f:
            job_data = eval(f.read())

        # Unpack args and make files for easy access
        ENV_NAME = job_data['env_name']
        PICKLE_FILE = OUT_DIR + '/trajectories.pickle'
        EXP_FILE = OUT_DIR + '/job_data.json'
        SEED = job_data['seed']

        # create env for visualization (dev)
        e=self.env_callable(**vis_env_kwargs)
        mean = np.zeros(e.action_dim)
        sigma = 1.0*np.ones(e.action_dim)
        filter_coefs = [sigma, job_data['filter']['beta_0'], job_data['filter']['beta_1'], job_data['filter']['beta_2']]
        trajectories = []

        ts=timer.time()
        start_time = timer.time()
        seed = job_data['seed']
        e.reset(seed=seed)

        agent = MPPI(e,
                     H=job_data['plan_horizon'],
                     paths_per_cpu=job_data['paths_per_cpu'],
                     num_cpu=job_data['num_cpu'],
                     kappa=job_data['kappa'],
                     gamma=job_data['gamma'],
                     mean=mean,
                     filter_coefs=filter_coefs,
                     default_act=job_data['default_act'],
                     seed=seed,
                     env_callable=self.env_callable,
                     env_kwargs=rollout_env_kwargs)

        for t in range(job_data['H_total']):
            agent.train_step(job_data['num_iter'])

        end_time = timer.time()
        print("Trajectory reward = %f" % np.sum(agent.sol_reward))
        print("Optimization time for this trajectory = %f" % (end_time - start_time))
        return agent.act_sequence

if __name__ == "__main__":

    planner = mppiPlanner()

    start_pose_file = open("/home/abba/msu_ws/src/motor_skills/motor_skills/envs/mj_jaco/assets/MjJacoDoorGrasps", 'rb')
    start_poses = pickle.load(start_pose_file)
    start=start_poses[8][:6]

    goal_pos = [0.0, 0.5, 0.5]
    goal_quat = [1, 0, 0, 0]
    # goal_pos = [0.22301166, 0.16430212, 0.43435462]
    # goal_quat = [ 9.75998480e-02,  2.66663423e-04,  6.52346744e-01, -7.51610220e-01]

    action_sequence = planner.plan(start, goal_pos, goal_quat)

    replay_env_kwargs = {
                  "vis" : False,
                  "n_steps" : int(1000),
                  "start" : start,
                  "goal_pos": goal_pos,
                  "goal_quat" : goal_quat}

    env_id=MjJacoMPPI
    e=env_id(**replay_env_kwargs)
    e.reset()
    start_state = e.get_env_state()

    np.save('my_trajectory.npy', action_sequence)

    H = len(action_sequence)
    print(H)

    obs=[]
    act=[]
    states=[]
    env_infos=[]
    rewards=[]

    for k in range(H):
        obs.append(e.get_obs())
        act.append(action_sequence[k])
        env_infos.append(e.get_env_infos())
        states.append(e.get_env_state())
        s, r, d, ifo = e.step(act[-1])
        rewards.append(r)

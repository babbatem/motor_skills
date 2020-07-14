import os
import time as timer
import numpy as np

import motor_skills
from motor_skills.envs.mj_jaco import MjJacoMPPI
from trajopt.envs.utils import get_environment
from trajopt.algos.mppi import MPPI

class mppiPlanner(object):
    """docstring for mppiPlanner."""
    def __init__(self):
        super(mppiPlanner, self).__init__()
        self.start = None
        self.goal_pos = None
        self.goal_quat = None
        self.env_callable = MjJacoMPPI
        self.env_kwargs = None

    def plan(self, start, goal_pos, goal_quat):

        env_kwargs = {"start" : start,
                      "goal_pos": goal_pos,
                      "goal_quat" : goal_quat}

        output = '/home/abba/msu_ws/src/motor_skills/motor_skills/outputs/foo'
        config = '/home/abba/msu_ws/src/motor_skills/motor_skills/configs/jaco_mppi_config.txt'

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

        e = get_environment(ENV_NAME)
        mean = np.zeros(e.action_dim)
        sigma = 1.0*np.ones(e.action_dim)
        filter_coefs = [sigma, job_data['filter']['beta_0'], job_data['filter']['beta_1'], job_data['filter']['beta_2']]
        trajectories = []

        ts=timer.time()
        for i in range(job_data['num_traj']):
            start_time = timer.time()
            print("Currently optimizing trajectory : %i" % i)
            seed = job_data['seed'] + i*12345
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
                         env_kwargs=self.env_kwargs)

            for t in trigger_tqdm(range(job_data['H_total']), VIZ):
                agent.train_step(job_data['num_iter'])

            end_time = timer.time()
            print("Trajectory reward = %f" % np.sum(agent.sol_reward))
            print("Optimization time for this trajectory = %f" % (end_time - start_time))
            trajectories.append(agent)
            pickle.dump(trajectories, open(PICKLE_FILE, 'wb'))

        print("Time for trajectory optimization = %f seconds" %(timer.time()-ts))
        pickle.dump(trajectories, open(PICKLE_FILE, 'wb'))

if __name__ == "__main__":

    planner = mppiPlanner()
    start = np.zeros(6)
    goal_pos = [0, 0.5, 0.5]
    goal_quat = [1, 0, 0, 0]
    planner.plan(start, goal_pos, goal_quat)

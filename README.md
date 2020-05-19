# motor_skills
controllers and environments!

# installation
1. install mujoco and mujoco_py using these [instructions](https://github.com/openai/mujoco-py). Note that we have a lab license and I will email you the access key.
2. python setup.py install

# usage
see door_test.py for minimal example; see motor_skills/core/mj_control.py for implementation of pd controller, to start.

# TODOs
1. add touch sensors to fingertips and read from them (maybe also FT sensor at wrist)
2. turn into gym environment (add reward function, properly inherit from gym.Env)
3. integrate learning algorithm and neural net (likely from DAPG/mjrl)
4. iterate on architecture until we're happy
5. wrap ROS around the learned controller(s)
6. wrap CIP representation around controller(s)

Skills we want to manually code:
1. free space motion (moveit)
2. grasping  (given poses from GPD, close the gripper)

Skills we want to learn:
1. door opening
2. door closing
3. ??

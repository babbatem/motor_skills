# motor_skills
controllers and environments!

# installation
1. install mujoco and mujoco_py using these [instructions](https://github.com/openai/mujoco-py). Note that we have a lab license.
2. python setup.py install (to get imports right; optionally add to pythonpath)
3. pip install -r reqs.txt
4. TODO: ompl installation instructions. At the moment the motion planner isn't plugged in.   

The gym env inheritance structure is wack. Here's a sketch, in the event that you need to dig into that code.

                MjJacoDoor  
                     |  
              MjJacoDoorCIPBase  
                /           \  
               /             \  
MjJacoDoorImpedanceCIP      MjJacoDoorImpedanceNaive       


See ```motor_skills/envs``` for gym environments.
See ```motor_skills/cip``` for abstract classes, particular implementations, VICES controller.
See ```motor_skills/core``` for control bits, e.g. PD control, operational space stuff.  

Perhaps the classifier (bandit!) should live in ```cip```. 

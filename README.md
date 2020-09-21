# motor_skills
controllers and environments!

# installation
1. install mujoco and mujoco_py using these [instructions](https://github.com/openai/mujoco-py). Note that we have a lab license.
2. python setup.py install (to get imports right; optionally add to pythonpath)
3. pip install -r reqs.txt (assumes python3)
4. TODO: ompl installation instructions. At the moment the motion planner isn't plugged in.   

See ```motor_skills/envs``` for gym environments.   
See ```motor_skills/cip``` for abstract classes, particular implementations, VICES controller.  
See ```motor_skills/core``` for control bits, e.g. PD control, operational space stuff.    

I think your first stop should be ```motor_skills/cip/cip.py```. It outlines the basic methods we expect a CIP to have. There's an ```ImpedanceCIP``` class which uses the VICES controller to map actions to joint torques. Each gym environment maintains a CIP, and calls a ```get_action``` method to convert from the environment's action space to torque. There is a particular CIP for the simulated test problem: ```MjDoorCIP.py```. It subclasses ```ImpedanceCIP```, and has a head that is implemented by ```MjGraspHead```. For the initiation set learning bit: the ```cip``` methods ```update_init_set``` and ```sample_init_set``` are there in the base class.

I can generate some data with which to train a classifier after learning is over. Eventually I think we want to generate data for an online classifier/bandit-kinda thing, we'll have to pass data up through the environment back to the learner. The learning code is mostly the same as last project's deep net experiments (see ```mjrl``` folder). If you look at ```mjrl/samplers/core.py``` you'll see the rollouts happening. We could log data in the ```paths``` dictionary, and collect it during training steps. I think we probably ought to implement our own learning algorithm (e.g. subclass ```mjrl/algos/batch_reinforce.py``` ala ```mjrl/algos/dapg``` and implement our own ```train_step``` method to also update the initiation, effect sets in addition to policy parameters).

Other TODOs: [link here forthcoming]  

The gym env inheritance structure is wack. Here's a sketch, in the event that you need to dig into that code.

                      MjJacoDoor    (loads the models, torque actions)  
                           |    
                    MjJacoDoorCIPBase  (overrides step function; abstract class)  
                      /           \    
                     /             \  
      MjJacoDoorImpedanceCIP      MjJacoDoorImpedanceNaive    
          (these two implement an ```init_cip``` method that loads the relevant controller)  

We might need to redesign this prior to releasing the code in any official capacity.

"""
Code for training
- bang-bang (using Genetic algorithm)
- indicators (using brute force)
"""

def bang_bang_training(params):
    """ Use genetic algorithm to evaluate best cut-offs """

def cross_validation(params):
    nhistory = 10
    data = "real"

    env = gym.make(
        "gym_examples/BatteryEnv-v0", 
        nhistory=nhistory, 
        data=date, 
        mode=params["env_mode"], 
        start_index = start_index,
        end_index = end_index,
        max_episode_steps=2000, # can change length here!"
    )


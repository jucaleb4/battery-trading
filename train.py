"""
Code for training
- bang-bang (using Genetic algorithm)
- indicators (using brute force)
"""
import numpy as np
import numpy.linalg as la

import genetic 

import gymnasium as gym
import gym_examples

def bang_bang_offline_training():
    """ Use genetic algorithm to evaluate best cut-offs """
    nhistory = 10
    data = "real"
    start_index = 0
    end_index = -1
    env_mode = "delay"

    # collect all the prices
    env = gym.make(
        "gym_examples/BatteryEnv-v0", 
        nhistory=nhistory, 
        data=data, 
        mode=env_mode, 
        start_index = start_index,
        end_index = end_index,
        max_episode_steps=76*4*24, # can change length here!"
    )
    env.reset()

    def objective(bounds):
        """ Evaluate profit with our strategy
        :param x: lower and upper bound for buy and sell price, respetictively
        """
        buy_price, sell_price = bounds
        if buy_price > sell_price:
            return 0

        sell, null, buy = 0, 1, 2
        s, _ = env.reset()
        last_lmp = s[1]
        total_rwd = 0
        while 1:
            if last_lmp >= sell_price:
                a = sell
            elif last_lmp <= buy_price:
                a = buy
            else:
                a = null
            s, r, term, trunc, _ = env.step(a)
            last_lmp = s[1]
            total_rwd += r
            if term or trunc:
                break
        return -total_rwd

    lbs, ubs = [-25, 0], [100, 200]
    x = genetic.optimize(objective, lbs, ubs, n_iter=100, n_pop=50, seed=None)

class OnlineBangBang():
    def __init__(

if __name__ == "__main__":
    bang_bang_offline_training()

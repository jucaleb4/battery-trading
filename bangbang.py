import os

import numpy as np
import numpy.linalg as la

import genetic
import gymnasium as gym
import gym_examples

from old_run import validate

def run_bangbang_offline(params):
    # computed by Genetic algorithm offline
    (buy_price, sell_price) = params["genetic_prices"] 
    sell, null, buy = 0, 1, 2

    def get_action(obs):
        last_lmp = obs["lmps"][0]
        if last_lmp >= sell_price:
            a = sell
        elif last_lmp <= buy_price:
            a = buy
        else:
            a = null
        return a

    fname = os.path.join("logs", f"alg=bang_offline_data=read_solar={params['solar_coloc']}.csv")
    params["fname"] = fname
    validate(get_action, params)

def run_bangbang_online(params):
    """ Use genetic algorithm to evaluate best cut-offs """
    nhistory = 16
    data = "real"
    start_index = 0 # 76*4*24,
    end_index = 76*4*24 # -1
    env_mode = "delay"

    # collect all the prices
    env = gym.make(
        "gym_examples/BatteryEnv-v0", 
        nhistory=nhistory, 
        data=data, 
        mode=env_mode, 
        max_episode_steps= 76*4*24, # 14*4*24, # can change length here!"
    )
    env.reset()

    def get_action(obs):
        
        def objective(bounds):
            """ Evaluate profit with our strategy
            :param x: lower and upper bound for buy and sell price, respetictively
            """
            buy_price, sell_price = bounds
            if buy_price > sell_price:
                return 0

            sell, null, buy = 0, 1, 2
            s, _ = env.reset(options={"start": start_idx})
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

def bang_bang_offline_training(env: gym.Env, max_iter, seed): 
    """ 
    Use genetic algorithm to evaluate best cut-offs 

    :param env: environment to train on.
    """
    print("Genetic algorithm does minimization, so we want to the smallest one")

    """
    nhistory = 10 # doesn't matter
    data = "real"
    start_index = 0
    end_index = (90-14)*4*24 # -1
    env_mode = "delay" # "delay"
    solar_coloc=False
    solar_scale=0.0
    print(f"Training with solar colocation: {solar_coloc}")

    # collect all the prices
    env = gym.make(
        "gym_examples/BatteryEnv-v0", 
        nhistory=nhistory, 
        start_index = start_index,
        end_index = end_index,
        max_episode_steps=end_index, # can change length here!"
        mode=env_mode, 
        daily_cost=0,
        more_data=False,
        delay_cost=False,
        solar_coloc=solar_coloc,
        solar_scale=solar_scale,
    )
    """

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
        last_lmp = s["lmps"][0]
        total_rwd = 0
        while 1:
            if last_lmp >= sell_price:
                a = sell
            elif last_lmp <= buy_price:
                a = buy
            else:
                a = null
            s, r, term, trunc, _ = env.step(a)
            last_lmp = s["lmps"][0]
            total_rwd += r
            if term or trunc:
                break
        # return negative since genetic minimizes
        return -total_rwd

    min_lmp = np.min(env.observation_space['lmps'].low)
    max_lmp = np.max(env.observation_space['lmps'].high)
    lbs, ubs = [min_lmp,]*2, [max_lmp,]*2
    bounds = genetic.optimize(
        objective, 
        lbs, 
        ubs, 
        n_iter=max_iter, 
        n_pop=50, 
        seed=seed
    )
    return bounds

if __name__ == "__main__":
    if input("train? (otherwise test): ").lower() in ["yes", "y"]:
        nhistory = 10 # doesn't matter
        data = "real"
        start_index = 0
        end_index = (90-14)*4*24 # -1
        env_mode = "delay" # "delay"
        solar_coloc=False
        solar_scale=0.0
        print(f"Training with solar colocation: {solar_coloc}")

        # collect all the prices
        env = gym.make(
            "gym_examples/BatteryEnv-v0", 
            nhistory=nhistory, 
            start_index = start_index,
            end_index = end_index,
            max_episode_steps=end_index, # can change length here!"
            mode=env_mode, 
            daily_cost=0,
            more_data=False,
            delay_cost=False,
            solar_coloc=solar_coloc,
            solar_scale=solar_scale,
        )
        bang_bang_offline_training(env, max_iter=32, seed=None)
    else:
        params = {
            "genetic_prices": (51.17, 82), # no solar
            # "genetic_prices": (50, 89),  # with solar
            # "genetic_prices": (48, 100.23), 
            "solar_coloc": False
        }   
        run_bangbang_offline(params)

import os
import time
import multiprocessing as mp

import numpy as np

# just for perlmutter
if True:
    import sys
    sys.path.append("/global/homes/c/cju33/.conda/envs/venv/lib/python3.12/site-packages")
    sys.path.append("/global/homes/c/cju33/gym-examples")

import argparse

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
import gymnasium as gym
import gym_examples
from gymnasium.wrappers import NormalizeObservation
from gymnasium.wrappers import NormalizeReward
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold

import warnings
import functools

from utils import TimelimitCallback

import tune
import bangbang

def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func

class SimpleLogger():
    def __init__(self, fname, mode):
        """ Saves data for: (soc, lmp, action, total_rwd) """
        self.fname = fname
        self.mode = mode
        self.data_arr = np.zeros((128, 4), dtype=float)
        self.ct = 0

    def store(self, data):
        """ Stores data
        :param data: tuple given as (obs, action, total_rwd)
        """
        (obs, a , total_rwd) = data

        (soc, lmp) = obs[0], obs[1]
        if self.mode == "qlearn":
            (soc, lmp) = (0,0)
        self.data_arr[self.ct] = (soc, lmp, a, total_rwd)
        self.ct += 1

        if self.ct == len(self.data_arr):
            self.data_arr = np.vstack((self.data_arr, np.zeros(self.data_arr.shape)))

    def save(self):
        fmt="%1.2f,%1.2f,%i,%1.2e"
        with open(self.fname, "wb") as fp:
            fp.write(b"soc,lmp,a,total_rwd\n")
            np.savetxt(fp, self.data_arr[:self.ct], fmt=fmt)
        print(f"Saved testing logs to {self.fname}")

def n_validate(env, n_steps, params, get_action):
    """ Test/validates policy.
    :param get_action: general function that takes in a current observation and outputs an action
    """
    obs = env.reset()
    n_cpu = obs.shape[0] if len(obs.shape) > 1 else 1
    action = get_action(obs)
    if len(action.shape) > 1:
        # TODO: Does this work?
        action_arr = np.zeros((n_steps, n_cpu, action.shape[1]))
    else:
        action_arr = np.zeros((n_steps, n_cpu))
    total_reward_arr = np.zeros((n_steps, n_cpu))
    try:
        obs = env.reset()
        total_reward = np.zeros(n_cpu, dtype=float)
        for t in range(n_steps):
            action = get_action(obs)
            # print(f"run.py [{t}]: ", action)
            (obs, reward, terminated, info) = env.step(action)
            # TODO: hacky, need more automatied way to do this
            total_reward += reward
            action_arr[t] = action
            total_reward_arr[t] = total_reward
            if np.any(terminated):
                obs = env.reset()

        if len(params.get('fname', '')) > 0:
            for i in range(n_cpu):
                fname = f"{params['fname']}_seed={params['seed']+i}.csv"
                logger = SimpleLogger(fname, params["env_mode"])
                for t in range(n_steps):
                    logger.store((obs[i], action_arr[t,i], total_reward_arr[t,i]))
                logger.save()

        final_rewards = total_reward_arr[-1,:]
        print(f"All final rewards: {final_rewards}")

        return final_rewards

    except KeyboardInterrupt:
        logger.save()

def run_n_qlearn(n_cpu, params):
    """ Runs multiple DQN experiments

    :param n: number of environments
    """
    assert "env_mode" in params
    assert "train_len" in params
    assert "policy_type" in params
    assert "exploration_fraction" in params
    assert "learning_starts" in params
    assert "batch_size" in params
    assert "learning_rate" in params
    assert "target_update_interval" in params

    params["start_index"] = 0 
    params["end_index"] = 4*24*76
    params["nhistory"] = 16

    nhistory = 16
    train_horizon = 76*4*24

    def make_env(rank: int, params={}, seed: int=0):
        def _init() -> gym.Env:
            env = gym.make(
                "gym_examples/BatteryEnv-v0", 
                seed=params["seed"],
                nhistory=params["nhistory"], 
                start_index=params["start_index"],
                end_index=params["end_index"],
                max_episode_steps=params["end_index"]-params["start_index"],
                mode=params["env_mode"], 
                daily_cost=params["daily_cost"],
                more_data=True,
                delay_cost=params.get("delay_cost", True),
                solar_coloc=params.get("solar_coloc", False)
            )
            env.reset(seed=seed+rank)
            if params.get("norm_obs", False):
                env = NormalizeObservation(env)
            if params.get("norm_rwd", False):
                env = NormalizeReward(env)
            if params.get("norm_obs", False) or params.get("norm_rwd", False):
                for _ in range(1000):
                    env.step(env.action_space.sample())
            env.reset(seed=seed+rank)
            return env

        set_random_seed(seed)
        return _init

    print(f"Making {n_cpu} environments")
    s_time = time.time()
    env = SubprocVecEnv([make_env(i, params) for i in range(n_cpu)])
    print(f"Finishing making {n_cpu} environments (time={time.time()-s_time:.2f}s)\nStarting training")
    s_time = time.time()

    # train
    model = DQN(
        params["policy_type"], 
        env, 
        verbose=1, 
        learning_starts=params["learning_starts"],
        exploration_fraction=params["exploration_fraction"], 
        exploration_final_eps=0.05,
        gradient_steps=params["gradient_steps"],
        batch_size=params["batch_size"],
        learning_rate=params["learning_rate"],
        target_update_interval=params["target_update_interval"],
    )
    timelimit_check = TimelimitCallback()

    # Setup logging file and modify parameters for testing
    fname_base = f"alg=dqn_data=real_env_mode={params['env_mode']}"
    fname_base += f"_train_len={params['train_len']}_daily_cost={params['daily_cost']}"
    fname_base += f"_solar={params['solar_coloc']}"
    fname = os.path.join("logs", fname_base)
    params["fname"] = fname

    eval_params = params.copy()
    eval_params["start_index"] = eval_params["end_index"]
    eval_params["end_index"] = 4*24*90
    n_steps = eval_params["end_index"] - eval_params["start_index"]
    eval_params["norm_rwd"] = False
    eval_params["daily_cost"] = 0
    eval_params["delay_cost"] = False
    eval_env = SubprocVecEnv([make_env(i, eval_params) for i in range(n_cpu)])
    eval_callback = EvalCallback(eval_env, best_model_save_path='models',
                             log_path='logs', n_eval_episodes=10,
                             deterministic=True )
    # callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=200_000, verbose=1)
    # eval_callback = EvalCallback(
    #     eval_env, 
    #     log_path = "logs",
    #     n_eval_episodes=10,
    #     eval_freq=7296,
    #     callback_on_new_best=callback_on_best,
    # )

    # model.learn(total_timesteps=params["train_len"], log_interval=1, callback=timelimit_check)
    model.learn(total_timesteps=params["train_len"], log_interval=1, callback=eval_callback)
    print(f"Finished training (time={time.time()-s_time:.2f}s)")

    print("Importing best model")
    model = DQN.load("models/best_model.zip")
    print("Finishing importing best model")

    # validation
    def get_action(obs):
        return model.predict(obs, deterministic=True)[0]

    return n_validate(eval_env, n_steps, params, get_action)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--env_mode", type=str, default="default", choices=["default", "difference", "sigmoid"], help="Environment type")
    parser.add_argument("--train_len", type=int, default=int(1e4), help="Number of training steps")
    parser.add_argument("--seed", type=int, default=-1, help="Seed for DQN (-1 is None)")
    parser.add_argument("--n_trials", type=int, default=1, help="Number of seeds to run")
    parser.add_argument("--wandb_tune", action="store_true", help="Tune with wandb")

    parser.add_argument("--more_data", action="store_true", help="Get more data from environment")
    parser.add_argument("--norm_obs", action="store_true", help="Normalize rewards between [0,1]")
    parser.add_argument("--norm_rwd", action="store_true", help="Normalize rewards between [0,1]")
    parser.add_argument("--solar_coloc", action="store_true", help="Use solar colocation")
    parser.add_argument("--daily_cost", type=float, default=0, help="Fixed cost every step applied during training")

    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--gradient_steps", type=float, default=-1, help="Gradient updates")
    parser.add_argument("--exploration_fraction", type=float, default=0.99, help="Exploration")
    params = vars(parser.parse_args())

    params["policy_type"] = "MlpPolicy"
    params["learning_starts"] = 20
    params["batch_size"] = 48
    params["target_update_interval"] = 540
    params["max_grad_norm"] = 10

    if params["seed"] < 0:
        params["seed"] = None
    
    if params["wandb_tune"]:
        tune.run_wandb(params)
    else:
        n_cpu = 1
        seed_0 = params["seed"] if params["seed"] != None else 0
        for s in range(seed_0, seed_0+ params["n_trials"]):
            params["seed"] = s
            run_n_qlearn(1, params)
        # run_bangbang_offline(params)

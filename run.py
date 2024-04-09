import os
import time
import multiprocessing as mp
import json

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
from gymnasium.wrappers import FlattenObservation
from gymnasium.wrappers import NormalizeObservation
from gymnasium.wrappers import NormalizeReward
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold

import warnings
import functools

from utils import TimelimitCallback
from utils import SimpleLogger

import tune
# import bangbang

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

def validate(env, n_steps, params, get_action):
    """ Test/validates policy.
    :param get_action: general function that takes in a current observation and outputs an action
    """
    obs, info = env.reset()
    all_info_keys = list(info.keys())
    info_keys = ["solar_reward", "grid_reward", "soc", "curr_lmp"]
    assert set(info_keys) <= set(all_info_keys)

    assert len(params.get('fname', '')) > 5
    fname = f"{params['fname']}_seed={params['seed']}.csv"
    logger = SimpleLogger(fname, info_keys)

    try:
        obs, info = env.reset()
        total_reward = 0
        for t in range(n_steps):
            action = get_action(obs)
            (obs, reward, terminated, truncated, info) = env.step(action)
            done = terminated or truncated
            total_reward += reward
            logger.store((info, action, total_reward))
            if done:
                obs, info = env.reset()

        logger.save()

        print(f"Final reward: {total_reward}")

        return total_reward

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
                more_data=params["more_data"],
                delay_cost=params.get("delay_cost", True),
                solar_coloc=params.get("solar_coloc", False),
                solar_scale=params.get("solar_scale", 0.0),
            )
            obs, info = env.reset()
            env = FlattenObservation(env)
            obs, info = env.reset(seed=seed+rank)
            if params.get("norm_obs", False):
                env = NormalizeObservation(env)
            if params.get("norm_rwd", False):
                env = NormalizeReward(env)
            if params.get("norm_obs", False) or params.get("norm_rwd", False):
                for _ in range(1000):
                    env.step(env.action_space.sample())
            obs, info = env.reset(seed=seed+rank)
            return env

        set_random_seed(seed)
        return _init

    # Setup logging file and modify parameters for testing
    print(f"Making environments")
    s_time = time.time()
    env = make_env(0, params)()
    test_env = make_env(1000, params)()
    setup_time = time.time()-s_time
    print(f"Setup time (time={setup_time:.2f}s)\nStarting training")
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

    # we should validate on the test environment
    eval_callback = EvalCallback(test_env, best_model_save_path='models',
                                log_path='logs', 
                                n_eval_episodes=3,
                                eval_freq=7296,
                                deterministic=True )
    model.learn(total_timesteps=params["train_len"], log_interval=1, callback=eval_callback)
    print(f"Finished training (time={time.time()-s_time:.2f}s)")

    print("Importing best model")
    model = DQN.load("models/best_model")
    print("Finishing importing best model")

    # validation
    def get_action(obs):
        # returns (action (as array), state)
        return model.predict(obs, deterministic=True)[0].flat[0]

    eval_params = params.copy()
    eval_params["start_index"] = eval_params["end_index"]
    eval_params["end_index"] = 4*24*90
    n_steps = eval_params["end_index"] - eval_params["start_index"]
    eval_params["norm_rwd"] = False
    eval_params["daily_cost"] = 0
    eval_params["delay_cost"] = False

    if len(params.get("settings_file", "")) > 5:
        settings_fname_raw = os.path.os.path.splitext(os.path.basename(params['settings_file']))[0]
        fname_base = f"alg=dqn_data=real_settings={settings_fname_raw}"
    else:
        raise Exception("You need to input a valid settings file")
    fname = os.path.join("logs", fname_base)
    eval_params["fname"] = fname

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
                more_data=params["more_data"],
                delay_cost=params.get("delay_cost", True),
                solar_coloc=params.get("solar_coloc", False),
                solar_scale=params.get("solar_scale_test", -1),
    eval_env = make_env(0, eval_params)()

    validate(eval_env, n_steps, eval_params, get_action)

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
    parser.add_argument("--solar_scale", type=float, default=0., help="Solar scaling")
    parser.add_argument("--solar_scale_test", type=float, default=-1, help="Solar scaling for testing")
    parser.add_argument("--daily_cost", type=float, default=0, help="Fixed cost every step applied during training")

    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--gradient_steps", type=float, default=-1, help="Gradient updates")
    parser.add_argument("--exploration_fraction", type=float, default=0.99, help="Exploration")

    parser.add_argument("--settings_file", type=str, default="")
    params = vars(parser.parse_args())

    params["policy_type"] = "MlpPolicy"
    params["learning_starts"] = 20
    params["batch_size"] = 48
    params["target_update_interval"] = 540
    params["max_grad_norm"] = 10
    params["solar_scale_test"] = params["solar_scale_test"] if params["solar_scale_test"] >= 0 else params["solar_scale"]
    if params["seed"] < 0:
        params["seed"] = None

    if len(params["settings_file"]) > 5 and params["settings_file"][-4:] == "json":
        with open(params["settings_file"], "r") as fp:
            new_settings = json.load(fp)
        params.update(new_settings)

    if params["wandb_tune"]:
        tune.run_wandb(params)
    else:
        n_cpu = 1
        seed_0 = params["seed"] if params["seed"] != None else 0
        for s in range(seed_0, seed_0+ params["n_trials"]):
            params["seed"] = s
            run_n_qlearn(1, params)
        # run_bangbang_offline(params)

import os
import time
import multiprocessing as mp
import concurrent
import json

import numpy as np

# just for perlmutter
if True:
    import sys
    sys.path.append("/global/homes/c/cju33/.conda/envs/venv/lib/python3.12/site-packages")
    sys.path.append("/global/homes/c/cju33/gym-examples")

import argparse

from stable_baselines3 import DQN
from stable_baselines3 import PPO
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

from bangbang import bang_bang_offline_training
from perfect import perfect_realistic

# import tune
# import bangbang

get_index = lambda x : 4*24*x

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

def process_battery_env(env, seed, norm_obs, norm_rwd):
    """ 
    Flattens battery's Dict environment and applys observation and reward
    normalization as needed. """
    env = FlattenObservation(env)
    env.reset(seed=seed)
    if norm_obs:
        env = NormalizeObservation(env)
    if norm_rwd:
        env = NormalizeReward(env)
    if norm_obs or norm_rwd:
        for _ in range(1000):
            env.step(env.action_space.sample())
    env.reset(seed=seed)
    return env

def validate(env, log_file, get_action):
    """ Test/validates policy.
    :param get_action: general function that takes in a current observation and outputs an action
    """
    obs, info = env.reset()
    all_info_keys = list(info.keys())
    info_keys = ["solar_reward", "grid_battery_reward", "solar_battery_reward", "soc", "soc_grid", "soc_solar", "curr_lmp", "net_load"]
    assert set(info_keys) <= set(all_info_keys)

    logger = SimpleLogger(log_file, info_keys)

    try:
        obs, info = env.reset()
        total_reward = 0
        while 1:
            action = get_action(obs)
            (obs, reward, terminated, truncated, info) = env.step(action)
            done = terminated or truncated
            total_reward += reward
            logger.store((info, action, total_reward))
            if done:
                break

        logger.save()

        print(f"Final reward: {total_reward}")

        return total_reward

    except KeyboardInterrupt:
        logger.save()

def get_train_and_test_envs(
        pnode_id: str,
        seed: int,
        n_history: int,
        # season: str,
        # NEW:
        train_season: str,
        test_season: str,
        train_start_date: int,
        train_len_dates: int,
        test_start_date: int,
        test_len_dates: int,
        env_mode: str,
        norm_obs: bool,
        norm_rwd: bool,
        more_data: bool,
        daily_cost: float,
        delay_cost: float,
        solar_coloc: bool,
        solar_scale: bool,
        solar_scale_test: bool,
        reset_mode: str='zero',
        reset_offset: int=0,
        preprocess_env: bool=False,
):

    # Setup logging file and modify parameters for testing
    print(f"Making environments")
    s_time = time.time()
    env = gym.make(
        id="gym_examples/BatteryEnv-v0", 
        pnode_id=pnode_id,
        nhistory=n_history, 
        season=train_season,
        index_offset=get_index(train_start_date),
        max_episode_steps=get_index(train_len_dates),
        mode=env_mode,
        daily_cost=daily_cost,
        more_data=more_data,
        delay_cost=delay_cost,
        solar_coloc=solar_coloc,
        solar_scale=solar_scale,
        seed=seed,
        reset_mode=reset_mode,
        reset_offset=reset_offset,
    )
    if preprocess_env:
        env = process_battery_env(env, seed, norm_obs, norm_rwd)

    test_env = gym.make(
        id="gym_examples/BatteryEnv-v0", 
        pnode_id=pnode_id,
        nhistory=n_history, 
        season=train_season,
        index_offset=get_index(train_start_date),
        max_episode_steps=get_index(train_len_dates),
        mode=env_mode,
        daily_cost=daily_cost,
        more_data=more_data,
        delay_cost=delay_cost,
        solar_coloc=solar_coloc,
        solar_scale=solar_scale,
        seed=1000+seed,
        reset_mode=None,
        reset_offset=0,
    )
    if preprocess_env:
        test_env = process_battery_env(test_env, 1000+seed, norm_obs, norm_rwd)

    eval_env = gym.make(
        id="gym_examples/BatteryEnv-v0", 
        pnode_id=pnode_id,
        nhistory=n_history, 
        season=test_season,
        index_offset=get_index(test_start_date),
        # end_index=get_index(test_end_date),
        max_episode_steps=get_index(test_len_dates),
        mode=env_mode,
        daily_cost=0,
        more_data=more_data,
        delay_cost=False,
        solar_coloc=solar_coloc,
        solar_scale=solar_scale_test,
        seed=seed,
        reset_mode=None,
        reset_offset=0,
    )
    if preprocess_env:
        eval_env = process_battery_env(eval_env, 2000+seed, norm_obs, norm_rwd=False)

    setup_time = time.time()-s_time
    print(f"Setup time (time={setup_time:.2f}s)\nStarting training")

    return env, test_env, eval_env

def run_qlearn(
        pnode_id: str, 
        seed: int,
        n_history: int,
        train_season: str,
        test_season: str,
        train_start_date: int,
        train_len_dates: int,
        test_start_date: int,
        test_len_dates: int,
        max_steps: int,
        env_mode: str,
        norm_obs: bool,
        norm_rwd: bool,
        more_data: bool,
        daily_cost: float,
        delay_cost: float,
        solar_coloc: bool,
        solar_scale: bool,
        solar_scale_test: bool,
        policy_type: str,
        learning_rate: float,
        max_grad_norm: float,
        learning_starts: int,
        exploration_fraction: float,
        exploration_final_eps: float,
        gradient_steps: int,
        batch_size: int,
        target_update_interval: int,
        log_folder: str,
    ):
    """ Runs multiple DQN experiments

    :param seed: 
    """
    env, test_env, eval_env = get_train_and_test_envs(
        pnode_id,
        seed,
        n_history,
        train_season,
        test_season,
        train_start_date,
        train_len_dates,
        test_start_date,
        test_len_dates,
        env_mode,
        norm_obs,
        norm_rwd,
        more_data,
        daily_cost,
        delay_cost,
        solar_coloc,
        solar_scale,
        solar_scale_test,
        preprocess_env=True,
    )

    # train
    s_time = time.time()
    model = DQN(
        policy=policy_type, 
        env=env, 
        verbose=1, 
        seed=seed,
        learning_starts=learning_starts,
        exploration_fraction=exploration_fraction, 
        exploration_final_eps=exploration_final_eps,
        gradient_steps=gradient_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        target_update_interval=target_update_interval,
    )
    # we should validate on the test environment
    # remove filename to get the same filepath

    test_best_callback = EvalCallback(
        test_env, 
        best_model_save_path=log_folder,
        log_path='logs', 
        n_eval_episodes=3,
        eval_freq=get_index(train_len_dates),
        deterministic=True
    )
    model.learn(total_timesteps=max_steps, log_interval=1, callback=test_best_callback)
    print(f"Finished training (time={time.time()-s_time:.2f}s)")

    print("Importing best model")
    model = DQN.load("%s/best_model" % log_folder)
    print("Finishing importing best model")

    # validation
    def get_action(obs):
        # returns (action (as array), state)
        return model.predict(obs, deterministic=True)[0].flat[0]

    log_file = os.path.join(log_folder, "seed=%s.csv" % seed)
    validate(eval_env, log_file, get_action)

def run_ppo(
        pnode_id: str, 
        seed: int,
        n_history: int,
        season: str,
        max_steps: int,
        env_mode: str,
        norm_obs: bool,
        norm_rwd: bool,
        more_data: bool,
        daily_cost: float,
        delay_cost: float,
        solar_coloc: bool,
        solar_scale: bool,
        solar_scale_test: bool,
        reset_mode: str,
        reset_offset: int,
        policy_type: str,
        n_steps: int,
        batch_size: int, 
        n_epochs: int,
        gamma: float,
        gae_lambda: float,
        normalize_advantage: bool,
        max_grad_norm: float,
        log_folder: str,
    ):
    """ Runs multiple DQN experiments

    :param seed: 
    """
    env, test_env, eval_env = get_train_and_test_envs(
        pnode_id,
        seed,
        n_history,
        season,
        env_mode,
        norm_obs,
        norm_rwd,
        more_data,
        daily_cost,
        delay_cost,
        solar_coloc,
        solar_scale,
        solar_scale_test,
        reset_mode,
        reset_offset,
        preprocess_env=True,
    )

    # train
    s_time = time.time()
    model = PPO(
        policy=policy_type, 
        env=env, 
        verbose=1, 
        seed=seed,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        normalize_advantage=normalize_advantage,
        max_grad_norm=max_grad_norm,
    )
    # we should validate on the test environment
    # remove filename to get the same filepath
    start_date = 0
    end_date = 90-14

    test_best_callback = EvalCallback(
        test_env, 
        best_model_save_path=log_folder,
        log_path='logs', 
        n_eval_episodes=3,
        eval_freq=get_index(end_date)-get_index(start_date),
        deterministic=True
    )
    model.learn(total_timesteps=max_steps, log_interval=1, callback=test_best_callback)
    print(f"Finished training (time={time.time()-s_time:.2f}s)")

    print("Importing best model")
    model = PPO.load("%s/best_model" % log_folder)
    print("Finishing importing best model")

    # validation
    def get_action(obs):
        # returns (action (as array), state)
        return model.predict(obs, deterministic=True)[0].flat[0]

    log_file = os.path.join(log_folder, "seed=%s.csv" % seed)
    validate(eval_env, log_file, get_action)

def run_bangbang(
        pnode_id: str,
        seed: int,
        max_iters: int,
        train_season: str,
        test_season: str,
        train_start_date: int,
        train_len_dates: int,
        test_start_date: int,
        test_len_dates: int,
        solar_coloc: bool,
        solar_scale: bool,
        solar_scale_test: bool,
        log_folder: str,
    ):
    """ Runs multiple DQN experiments

    :param seed: 
    """
    env, _, eval_env = get_train_and_test_envs(
        pnode_id,
        seed,
        n_history=4,
        train_season=train_season,
        test_season=test_season,
        train_start_date=train_start_date,
        train_len_dates=train_len_dates,
        test_start_date=test_start_date,
        test_len_dates=test_len_dates,
        env_mode="default",
        norm_obs=False,
        norm_rwd=False,
        more_data=False,
        daily_cost=0,
        delay_cost=False,
        solar_coloc=solar_coloc,
        solar_scale=solar_scale,
        solar_scale_test=solar_scale_test,
        preprocess_env=False,
    )

    # train
    buy_price, sell_price = bang_bang_offline_training(env, max_iter=max_iters, seed=seed)

    # validation
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

    log_file = os.path.join(log_folder, "seed=%s.csv" % seed)
    validate(eval_env, log_file, get_action)

def run_onlysell(
        pnode_id: str,
        seed: int,
        train_season: str,
        test_season: str,
        train_start_date: int,
        train_len_dates: int,
        test_start_date: int,
        test_len_dates: int,
        solar_coloc: bool,
        solar_scale: bool,
        solar_scale_test: bool,
        log_folder: str,
    ):
    """ Runs multiple DQN experiments

    :param seed: 
    """
    env, _, eval_env = get_train_and_test_envs(
        pnode_id,
        seed,
        n_history=4,
        train_season=train_season,
        test_season=test_season,
        train_start_date=train_start_date,
        train_len_dates=train_len_dates,
        test_start_date=test_start_date,
        test_len_dates=test_len_dates,
        env_mode="default",
        norm_obs=False,
        norm_rwd=False,
        more_data=False,
        daily_cost=0,
        delay_cost=False,
        solar_coloc=solar_coloc,
        solar_scale=solar_scale,
        solar_scale_test=solar_scale_test,
        preprocess_env=False,
    )

    # validation
    sell, null, buy = 0, 1, 2
    get_action = lambda obs : sell

    log_file = os.path.join(log_folder, "seed=%s.csv" % seed)
    validate(eval_env, log_file, get_action)

def _run(settings):
    seed_0 = settings["seed"] 
    for seed in range(seed_0, seed_0+settings["max_trials"]):
        if settings['alg'] == 'qlearn':
            run_qlearn(
                settings["pnode_id"],
                seed,
                settings["n_history"],
                settings["train_season"],
                settings["test_season"],
                settings["train_start_date"],
                settings["train_len_dates"],
                settings["test_start_date"],
                settings["test_len_dates"],
                settings["max_steps"],
                settings["env_mode"],
                settings["norm_obs"],
                settings["norm_rwd"],
                settings["more_data"],
                settings["daily_cost"],
                settings["delay_cost"],
                settings["solar_coloc"],
                settings["solar_scale"],
                settings["solar_scale_test"],
                settings["policy_type"],
                settings["learning_rate"],
                settings["max_grad_norm"],
                settings["learning_starts"],
                settings["exploration_fraction"],
                settings["exploration_final_eps"],
                settings["gradient_steps"],
                settings["batch_size"],
                settings["target_update_interval"],
                settings["log_folder"],
            )
        elif settings['alg'] == 'ppo':
            run_ppo(
                settings['pnode_id'],
                seed,
                settings['n_history'],
                settings['season'],
                settings['max_steps'],
                settings['env_mode'],
                settings['norm_obs'],
                settings['norm_rwd'],
                settings['more_data'],
                settings['daily_cost'],
                settings['delay_cost'],
                settings['solar_coloc'],
                settings['solar_scale'],
                settings['solar_scale_test'],
                settings['reset_mode'],
                settings['reset_offset'],
                settings['policy_type'],
                settings['n_steps'],
                settings['batch_size'],
                settings['n_epochs'],
                settings['gamma'],
                settings['gae_lambda'],
                settings['normalize_advantage'],
                settings['max_grad_norm'],
                settings['log_folder'],
            )
        elif settings['alg'] == 'bangbang':
            run_bangbang(
                settings["pnode_id"],
                seed,
                settings['max_iters'],
                settings["train_season"],
                settings["test_season"],
                settings["train_start_date"],
                settings["train_len_dates"],
                settings["test_start_date"],
                settings["test_len_dates"],
                settings['solar_coloc'],
                settings['solar_scale'],
                settings['solar_scale_test'],
                settings["log_folder"],
            )
        elif settings['alg'] = 'milp':
            perfect_realistic(
                settings['pnode_id'],
                settings['test_season'], 
                settings["test_start_date"],
                settings["test_len_dates"],
                settings['solar_scale']
                settings['log_folder'],
        elif settings['alg'] == 'onlysell':
            run_onlysell(
                settings["pnode_id"],
                seed,
                settings["train_season"],
                settings["test_season"],
                settings["train_start_date"],
                settings["train_len_dates"],
                settings["test_start_date"],
                settings["test_len_dates"],
                settings['solar_coloc'],
                settings['solar_scale'],
                settings['solar_scale_test'],
                settings["log_folder"],
            )
        else:
            raise Exception("Unknown alg %s" % settings['alg'])

def read_and_run(i):
    settings_file = os.path.join("settings", "04_13_2024", "exp_0", "run_%s.json" % i)
    with open(settings_file, "r") as fp:
        settings = json.load(fp)
    _run(settings)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    """
    parser.add_argument("--env_mode", type=str, default="default", choices=["default", "difference", "sigmoid"], help="Environment type")
    parser.add_argument("--train_len", type=int, default=int(1e4), help="Number of training steps")
    parser.add_argument("--seed", type=int, default=-1, help="Seed for DQN (-1 is None)")
    parser.add_argument("--max_trials", type=int, default=1, help="Number of seeds to run")
    parser.add_argument("--wandb_tune", action="store_true", help="Tune with wandb")

    parser.add_argument("--more_data", action="store_true", help="Get more data from environment")
    parser.add_argument("--norm_obs", action="store_true", help="Normalize rewards between [0,1]")
    parser.add_argument("--norm_rwd", action="store_true", help="Normalize rewards between [0,1]")
    parser.add_argument("--solar_coloc", action="store_true", help="Use solar colocation")
    parser.add_argument("--solar_scale", type=float, default=0., help="Solar scaling")
    parser.add_argument("--solar_scale_test", type=float, default=-1, help="Solar scaling for testing")
    parser.add_argument("--daily_cost", type=float, default=0, help="Fixed cost every step applied during training")
    parser.add_argument("--delay_cost", action="store_true", help="Delay cost of buying")

    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--gradient_steps", type=float, default=-1, help="Gradient updates")
    parser.add_argument("--exploration_fraction", type=float, default=0.99, help="Exploration")
    """

    parser.add_argument("--settings", type=str)
    parser.add_argument("--parallel", action="store_true")
    args = parser.parse_args()

    if len(args.settings) > 5 and args.settings[-4:] == "json":
        with open(args.settings, "r") as fp:
            settings = json.load(fp)
    else:
        raise Exception("Invalid settings file args.settings")

    _run(settings)

    # parallel does not work for SB3, probably need to use CleanRL
    """
    if not args.parallel:
        if len(args.settings) > 5 and args.settings[-4:] == "json":
            with open(args.settings, "r") as fp:
                settings = json.load(fp)
        else:
            raise Exception("Invalid settings file args.settings")

        _run(settings)

    else:
        n_experiments = 19
        n_cpus = mp.cpu_count()
        print("num cpus: %s" % n_cpus)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            for i in range(n_experiments):
                executor.submit(read_and_run, i)
    """

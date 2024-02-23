import os
import multiprocessing as mp

import numpy as np

# just for perlmutter
if True:
    import sys
    sys.path.append("/global/homes/c/cju33/.conda/envs/venv/lib/python3.12/site-packages")
    sys.path.append("/global/homes/c/cju33/gym-examples")

import argparse

import gymnasium as gym
import gym_examples

from stable_baselines3 import DQN

class SimpleLogger():
    def __init__(self, fname, mode, obs_scale=1, obs_shift=0, rwd_scale=1):
        """ Saves data for: (soc, lmp, action, total_rwd) """
        self.fname = fname
        self.mode = mode
        self.data_arr = np.zeros((128, 4), dtype=float)
        self.ct = 0
        self.obs_scale = obs_scale
        self.obs_scale = obs_scale

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

def run_exp(params):
    """ Runs DQN experiment 

    :param seed: seed of DQN
    """
    assert "env_mode" in params
    assert "seed" in params
    assert "train_len" in params

    fname = os.path.join(
        "logs", 
        f"alg=dqn_data=real_env_mode={params['env_mode']}_train_len={params['train_len']}_norm_obs={params['norm_obs']}_seed={params['seed']}.csv"
    )

    nhistory = 10
    train_horizon = 76*4*24
    env = gym.make(
        "gym_examples/BatteryEnv-v0", 
        nhistory=nhistory, 
        data="real", 
        end_index=train_horizon,
        max_episode_steps=train_horizon,
        mode=params["env_mode"], 
    )

    lows, highs = env.observation_space.low, env.observation_space.high
    obs_scale = np.reciprocal((highs - lows).astype("float"))
    obs_shift = -lows
    rwd_scale = 0.01
    if params.get("norm_obs", False):
        env = gym.wrappers.TransformObservation(env, lambda obs : np.multiply(obs + obs_shirt, obs_scale))
    env = gym.wrappers.TransformReward(env, lambda r : rwd_scale*r)
    logger = SimpleLogger(fname, params["env_mode"], obs_scale=obs_scale, obs_shift=obs_shift, rwd_scale=rwd_scale)

    model = DQN(
        "MlpPolicy", 
        env, 
        verbose=1, 
        seed=params["seed"],
        learning_starts=1024,
        exploration_fraction=1, # use less exploration
        exploration_final_eps=0.05,
        gradient_steps=1,
        batch_size=32,
        learning_rate=0.001,
    )
    model.learn(total_timesteps=params["train_len"], log_interval=12)

    # new environment for testing 
    env = gym.make(
        "gym_examples/BatteryEnv-v0", 
        nhistory=nhistory, 
        # data="periodic", 
        mode=params["env_mode"], 
        avoid_penalty=True,
    )
    env._max_episode_steps = 672
    lows, highs = env.observation_space.low, env.observation_space.high
    rng = highs - lows
    if params.get("norm_obs", False):
        env = gym.wrappers.TransformObservation(env, lambda obs : np.divide(obs - lows, rng))

    try:
        obs, info = env.reset()
        total_reward = 0
        for _ in range(1024):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            logger.store((obs, action, total_reward))
            if terminated or truncated:
                obs, info = env.reset()

        logger.save()
        print(f"Total reward: {total_reward}")

    except KeyboardInterrupt:
        logger.save()

def run_parallel_exp(params, num_exps):
    if num_exps == None or num_exps < 1:
        print("Need to pass in >= 1 experiments")
        exit(0)

    n_cores = mp.cpu_count()
    n_cores_to_use = int(n_cores/2)
    ps = [None] * n_cores_to_use
    ct = 0

    for seed in range(num_exps):
        if ct == n_cores_to_use:
            for p in ps:
                p.join()
            ct = 0

        params_i = params.copy()
        params_i["seed"] = seed
        p = mp.Process(target=run_exp, args=(params_i,))
        p.start()
        ps[ct] = p
        ct += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--env_mode", type=str, required=True, choices=["delay", "qlearn", "penalize_full", "penalize_wait"], help="Environment type")
    parser.add_argument("--train_len", type=int, default=int(1e4), help="Number of training steps")
    parser.add_argument("--seed", type=int, default=-1, help="Seed for DQN (-1 is None)")
    parser.add_argument("--parallel", action="store_true", help="Use multiprocessing to run experiments in parallel")

    parser.add_argument("--norm_obs", action="store_true", help="Normalize rewards between [0,1]")
    params = vars(parser.parse_args())

    if params["seed"] < 0:
        params["seed"] = None
    
    if params["parallel"]:
        run_parallel_exp(params, params["seed"])
    else:
        run_exp(params)

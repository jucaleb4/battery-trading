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
        self.obs_shift = obs_shift
        self.rwd_scale = rwd_scale

    def store(self, data):
        """ Stores data
        :param data: tuple given as (obs, action, total_rwd)
        """
        (obs, a , total_rwd) = data
        normalized_obs = np.divide(obs, self.obs_scale) - self.obs_shift
        rescaled_total_rwd = total_rwd/self.rwd_scale

        (soc, lmp) = normalized_obs[0], normalized_obs[1]
        if self.mode == "qlearn":
            (soc, lmp) = (0,0)
        self.data_arr[self.ct] = (soc, lmp, a, rescaled_total_rwd)
        self.ct += 1

        if self.ct == len(self.data_arr):
            self.data_arr = np.vstack((self.data_arr, np.zeros(self.data_arr.shape)))

    def save(self):
        fmt="%1.2f,%1.2f,%i,%1.2e"
        with open(self.fname, "wb") as fp:
            fp.write(b"soc,lmp,a,total_rwd\n")
            np.savetxt(fp, self.data_arr[:self.ct], fmt=fmt)
        print(f"Saved testing logs to {self.fname}")

def n_validate(env, n_steps, params, get_action, obs_scale, obs_shift):
    """ Test/validates policy.
    :param get_action: general function that takes in a current observation and outputs an action
    """
    obs, info = env.reset()
    n_cpu = obs.shape[0]
    action = get_action(obs)
    obs_arr = np.zeros((n_steps, n_cpu, obs.shape[1]))
    action_arr = np.zeros((n_steps, n_cpu, action.shape[1]))
    reward_arr = np.zeros((n_steps, n_cpu))
    try:
        obs, info = env.reset()
        total_reward = np.zeros(n_cpu, dtype=float)
        for t in range(n_steps):
            action = get_action(obs)
            (obs, reward, terminated, info) = env.step(int(action))
            total_reward += reward
            obs_arr[t] = obs
            action_arr[t] = action
            reward_arr[t] = reward
            if terminated:
                obs, info = env.reset()

        for i in range(n_cpu):
            fname = f"{params['fname']}_seed={i}.csv"
            logger = SimpleLogger(fnames, params["env_mode"])
            for t in range(n_steps):
                obs_rescaled = np.divide(obs_arr[t,i], obs_scale) - obs_shift
                logger.store((obs_rescaled, action_arr[t,i], reward_arr[t,i]))
            logger.save()

        final_rewards = reward_arr[-1,:]
        median_final_reward = np.median(final_rewards)
        print(f"Final median reward: {median_final_reward}")

    except KeyboardInterrupt:
        logger.save()

def get_normalized_env(env, params):
    lows, highs = env.observation_space.low, env.observation_space.high
    obs_scale = np.reciprocal((highs - lows).astype("float"))
    obs_shift = -lows
    rwd_scale = 0.01
    if params.get("norm_obs", False):
        env = gym.wrappers.TransformObservation(env, lambda obs : np.multiply(obs + obs_shift, obs_scale))
    env = gym.wrappers.TransformReward(env, lambda r : rwd_scale*r)
    return (env, obs_scale, obs_shift)

def run_qlearn(params):
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
        more_data=params["more_data"],
    )

    obs_scale, obs_shift, rwd_scale = 1, 0, 1
    if params.get("norm_obs", False):
        (env, obs_scale, obs_shift) = get_normalized_env(env, params)

    # train
    model = DQN(
        "MlpPolicy", 
        env, 
        verbose=1, 
        seed=params["seed"],
        learning_starts=100,
        exploration_fraction=0.99, # use less exploration
        exploration_final_eps=0.05,
        gradient_steps=-1,
        batch_size=32,
        learning_rate=0.001,
        target_update_interval=100,
    )
    model.learn(total_timesteps=params["train_len"], log_interval=1)

    # validation
    def get_action(obs):
        rescaled_obs = np.divide(obs - obs_shift, obs_scale)
        action, _ = model.predict(obs, deterministic=True)
        return action

    fname = os.path.join(
        "logs", 
        f"alg=dqn_data=real_env_mode={params['env_mode']}_train_len={params['train_len']}_norm_obs={params['norm_obs']}_more_data={params['more_data']}_seed={params['seed']}.csv"
    )
    params["fname"] = fname
    validate(params, get_action)

def run_n_qlearn(n_cpu, params):
    """ Runs multiple DQN experiments

    :param n: number of environments
    """
    assert "env_mode" in params
    assert "seed" in params
    assert "train_len" in params
    assert "policy_type" in params
    assert "exploration_fraction" in params
    assert "learning_starts" in params
    assert "batch_size" in params
    assert "learning_rate" in params
    assert "target_update_interval" in params
    assert n_cpu > 1, f"Requies n_cpu > 1 (currently n_cpu={n_cpu})"

    params["start_index"] = 0 
    params["end_index"] = 4*24*76
    params["nhistory"] = 10

    nhistory = 10
    train_horizon = 76*4*24

    # get normalizing factor (TODO: Is there a better way of doing this?)
    env = gym.make(
        "gym_examples/BatteryEnv-v0", 
        nhistory=params["nhistory"], 
        end_index=params["train_horizon"],
        max_episode_steps=params["train_horizon"],
        mode=params["env_mode"], 
        more_data=params["more_data"],
    )
    (_, obs_scale, obs_shift) = get_normalized_env(env, params)

    def make_env(env_id: str, rank: int, params={}, seed: int=0):

        def _init() -> gym.Env:
            env = gym.make(
                "gym_examples/BatteryEnv-v0", 
                nhistory=params["nhistory"], 
                start_index=params["start_index"],
                end_index=params["end_index"],
                max_episode_steps=params["end_index"]-params["start_index"],
                mode=params["env_mode"], 
                more_data=params["more_data"],
            )
            (env, obs_scale, obs_shift) = get_normalized_env(env, params)
            env.reset(seed=seed+rank)
            return env

        set_random_seed(seed)
        return _init

    print(f"Making {n_cpu} environments")
    s_time = time.time()
    env = SubprocVecEnv([make_env(env_id, i, params) for i in range(n_cpu)])
    print(f"Making {n_cpu} environments (time={time.time()-s_time:.2f}s)\nStarting training")

    # train
    model = DQN(
        params["policy_type"], 
        env, 
        verbose=1, 
        learning_starts=params["learning_starts"],
        exploration_fraction=params["exploration_fraction"], 
        exploration_final_eps=0.05,
        gradient_steps=-1,
        batch_size=params["batch_size"],
        learning_rate=params["learning_rate"],
        target_update_interval=params["target_update_interval"],
    )
    model.learn(total_timesteps=params["train_len"], log_interval=1)

    def get_action(obs):
        return model.predict(obs, deterministic=True)[0]

    fname_base = f"alg=dqn_data=real_env_mode={params['env_mode']}"
    fname_base += f"_train_len={params['train_len']}_norm_obs={params['norm_obs']}"
    fname_base += f"_more_data={params['more_data']}"
    fname = os.path.join("logs", fname_base)
    params["fname"] = fname

    params["start_index"] = params["end_index"]
    params["end_index"] = 4*24*90
    n_steps = params["end_index"] - params["start_index"]
    env = SubprocVecEnv([make_env(env_id, i, params) for i in range(n_cpu)])
    n_validate(env, n_steps, params, get_action, obs_scale, obs_shift)

def run_bangbang_offline(params):
    (buy_price, sell_price) = (54.11, 114) # computed by Genetic algorithm offline
    sell, null, buy = 0, 1, 2

    def get_action(obs):
        last_lmp = obs[1]
        if last_lmp >= sell_price:
            a = sell
        elif last_lmp <= buy_price:
            a = buy
        else:
            a = null
        return a

    fname = os.path.join("logs", f"alg=bang_offline_data=real_env_mode={params['env_mode']}.csv")
    params["fname"] = fname
    validate(params, get_action)

def run_bangbang_online(params):
    """ Use genetic algorithm to evaluate best cut-offs """
    nhistory = 10
    data = "real"
    start_index = 76*4*24,
    end_index = -1
    env_mode = "delay"

    # collect all the prices
    env = gym.make(
        "gym_examples/BatteryEnv-v0", 
        nhistory=nhistory, 
        data=data, 
        mode=env_mode, 
        max_episode_steps=14*4*24, # can change length here!"
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

def run_parallel_exp(params, num_exps):
    """
    # TODO: Make this run on Perlmutter 
    """
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
        p = mp.Process(target=run_qlearn, args=(params_i,))
        p.start()
        ps[ct] = p
        ct += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--env_mode", type=str, required=True, choices=["delay", "qlearn", "penalize_full", "penalize_wait"], help="Environment type")
    parser.add_argument("--train_len", type=int, default=int(1e4), help="Number of training steps")
    parser.add_argument("--seed", type=int, default=-1, help="Seed for DQN (-1 is None)")
    parser.add_argument("--parallel", action="store_true", help="Use multiprocessing to run experiments in parallel")
    parser.add_argument("--more_data", action="store_true", help="Get more data from environment")

    parser.add_argument("--norm_obs", action="store_true", help="Normalize rewards between [0,1]")
    params = vars(parser.parse_args())

    params["policy_type"] = "MlpPolicy"
    params["exploration_fraction"] = 0.99
    params["learning_starts"] = 100
    params["batch_size"] = 32
    params["learning_rate"] = 0.001
    params["target_update_interval"] = 100

    if params["seed"] < 0:
        params["seed"] = None
    
    if params["parallel"]:
        n_cpu = 4
        run_n_qlearn(n_cpu, params)
    else:
        run_qlearn(params)
        # run_bangbang_offline(params)

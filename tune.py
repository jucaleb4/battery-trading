import numpy as np
import wandb
from run import run_n_qlearn

def get_wandb_tuning_sweep_id(env_mode, daily_cost):
    sweep_config = {
        "method": "random",
    }

    metric = {
        'name': 'median_reward',
        'goal': 'maximize'
    }
    sweep_config['metric'] = metric

    parameters_dict = {
        'policy_type': { 
            'values': ['MlpPolicy'],
        },
        'train_freq': {
            'values': [4, (1, "episode")],
        },
        'exploration_fraction': {  # a flat distribution between 0 and 0.1
            'distribution': 'uniform',
            'min': 0.9,
            'max': 0.995,
        },
        'learning_starts': { # integers between 32 and 256 with evenly-distributed logarithms
            'distribution': 'q_log_uniform_values',
            'q': 10,
            'min': 10,
            'max': 1000,
        },
        'gradient_steps': { 
            'values': [-1, 1, 10, 100],
        },
        'batch_size': { 
            'distribution': 'q_log_uniform_values',
            'q': 4,
            'min': 16,
            'max': 256,
        },
        'learning_rate': { 
            'distribution': 'uniform',
            'min': 0,
            'max': 0.1
        },
        'target_update_interval': { 
            'distribution': 'q_log_uniform_values',
            'q': 10,
            'min': 10,
            'max': 1000,
        },
        'max_grad_norm': {
            'distribution': 'q_log_uniform_values',
            'q': 10,
            'min': 10,
            'max': 10000,
        },
    }
    sweep_config['parameters'] = parameters_dict

    sweep_id = wandb.sweep(sweep_config, project=f"battery-trading-rl-mode={env_mode}-daily_cost={daily_cost}")

    return sweep_id

def wandb_run_difference_daily(config=None):
    """ 
    Tunes QLearn with consecutive difference in LMP and forecasts and daily
    fixed cost of $100
    """

    # Initialize a new wandb run
    with wandb.init(config=config):
        config = wandb.config
        params = dict(config)
        params["train_len"] = 100000
        params["norm_rwd"] = True
        params["norm_obs"] = True
        params["env_mode"] = "difference"
        params["daily_cost"] = 100

        n_cpu = 1
        n_trials = 3
        final_rewards_arr = np.zeros(n_trials, dtype=float)

        for i in range(n_trials):
            params["seed"] = i
            final_reward = run_n_qlearn(n_cpu, params)
            final_rewards_arr[i] = np.median(final_reward)

        wandb.log({
            "median_reward": np.median(final_rewards_arr), 
            "all_rewards": final_rewards_arr,
        })    

def wandb_run_sigmoid_daily(config=None):
    """ 
    Tunes QLearn with consecutive difference transformed via sigmoid in LMP
    and forecasts and daily fixed cost of $100
    """

    # Initialize a new wandb run
    with wandb.init(config=config):
        config = wandb.config
        params = dict(config)
        params["train_len"] = 100000
        params["norm_rwd"] = True
        params["norm_obs"] = False
        params["env_mode"] = "sigmoid"
        params["daily_cost"] = 100

        n_cpu = 1
        n_trials = 3
        final_rewards_arr = np.zeros(n_trials, dtype=float)

        for i in range(n_trials):
            params["seed"] = i
            final_reward = run_n_qlearn(n_cpu, params)
            final_rewards_arr[i] = np.median(final_reward)

        wandb.log({
            "median_reward": np.median(final_rewards_arr), 
            "all_rewards": final_rewards_arr,
        })    

def wandb_run_sigmoid_free(config=None):
    """ 
    tunes qlearn with consecutive difference transformed via sigmoid in lmp
    and forecasts and no daily costs
    """

    # Initialize a new wandb run
    with wandb.init(config=config):
        config = wandb.config
        params = dict(config)
        params["train_len"] = 100000
        params["norm_rwd"] = True
        params["norm_obs"] = False
        params["env_mode"] = "sigmoid"
        params["daily_cost"] = 0

        n_cpu = 1
        n_trials = 3
        final_rewards_arr = np.zeros(n_trials, dtype=float)

        for i in range(n_trials):
            params["seed"] = i
            final_reward = run_n_qlearn(n_cpu, params)
            final_rewards_arr[i] = np.median(final_reward)

        wandb.log({
            "median_reward": np.median(final_rewards_arr), 
            "all_rewards": final_rewards_arr,
        })    

def wandb_run_default_daily(config=None):
    """ 
    Tunes qlearn with no transformation (just normalization) and no daily
    cost of $100
    """

    # Initialize a new wandb run
    with wandb.init(config=config):
        config = wandb.config
        params = dict(config)
        params["train_len"] = 100000
        params["norm_rwd"] = True
        params["norm_obs"] = True
        params["env_mode"] = "default"
        params["daily_cost"] = 100

        n_cpu = 1
        n_trials = 3
        final_rewards_arr = np.zeros(n_trials, dtype=float)

        for i in range(n_trials):
            params["seed"] = i
            final_reward = run_n_qlearn(n_cpu, params)
            final_rewards_arr[i] = np.median(final_reward)

        wandb.log({
            "median_reward": np.median(final_rewards_arr), 
            "all_rewards": final_rewards_arr,
        })    

def run_wandb(params):
    n_runs = 64
    sweep_id = get_wandb_tuning_sweep_id(params["env_mode"], params["daily_cost"])
    if params["env_mode"] == "difference":
        wandb.agent(sweep_id, wandb_run_difference_daily, count=n_runs)
    elif params["env_mode"] == "sigmoid" and params["daily_cost"] > 0:
        wandb.agent(sweep_id, wandb_run_sigmoid_daily, count=n_runs)
    elif params["env_mode"] == "sigmoid":
        wandb.agent(sweep_id, wandb_run_sigmoid_free, count=n_runs)
    else:
        wandb.agent(sweep_id, wandb_run_default_daily, count=n_runs)

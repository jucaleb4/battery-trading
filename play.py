import sys
import time
import multiprocessing as mp

sys.path.append("/global/homes/c/cju33/.conda/envs/venv/lib/python3.12/site-packages")
sys.path.append("/global/homes/c/cju33/gym-examples")

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
import gymnasium as gym
import gym_examples

def make_env(env_id: str, rank: int, seed: int=0):

    def _init() -> gym.Env:
        env = gym.make(env_id)
        env.reset(seed=seed+rank)
        return env

    set_random_seed(seed)
    return _init

def run(i, n_steps=1_000_000):
    print(f"id {i} setting up")
    s_time = time.time()
    n_history = 10

    env = gym.make(
        "gym_examples/BatteryEnv-v0", 
        nhistory=n_history, 
        max_episode_steps=1000, # can change length here!"
    )

    model = DQN(
        "MlpPolicy", 
        env, 
        verbose=1, 
        seed=i,
        learning_starts=100,
        exploration_fraction=0.99, # use less exploration
        exploration_final_eps=0.05,
        gradient_steps=-1,
        batch_size=32,
        learning_rate=0.001,
        target_update_interval=100,
    )
    print(f"id {i} running")
    model.learn(total_timesteps=2000, log_interval=1)
    print(f"id {i} finished in {time.time() - s_time:.2f}s")

def run_parallel(n_cpu, n_steps=1_000_000):
    print(f"setting up (n_cpu={n_cpu})")
    s_time = time.time()
    env_id = "gym_examples/BatteryEnv-v0"
    env = SubprocVecEnv([make_env(env_id, i) for i in range(n_cpu)])
    model = DQN(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_starts=100,
        exploration_fraction=0.99, # use less exploration
        exploration_final_eps=0.05,
        gradient_steps=-1,
        batch_size=32,
        learning_rate=0.001,
        target_update_interval=100,
    )
    
    print(f"running (n_cpu={n_cpu})")
    model.learn(total_timesteps=2000, log_interval=1)
    print(f"finished in {time.time() - s_time:.2f}s (n_cpu={n_cpu})")

if __name__ == "__main__":
    run(0)
    n_cpu = 10
    run_parallel(n_cpu)

import gymnasium as gym
import gym_examples

"""
Sanity check of environment so that we can make the money
"""

nhistory = 10
mode = "qlearn"
env = gym.make("gym_examples/BatteryEnv-v0", nhistory=nhistory, data="periodic", mode=mode)
env._max_episode_steps = 2048

a = 0
ndays = 4
env.reset()
total_reward = 0
for t in range(24*4*ndays):
    i = t % (4*24) 
    if 19 <= i <= 29:
        a = 0 # top of peak, so sell
    elif 67 <= i <= 75:
        a = 2
    else:
        a = 1 # bottom of peak, so buy
    obs, reward, terminated, truncated, info = env.step(a)
    total_reward += reward
    # print(t, total_reward, obs)
print(f"total_reward: {total_reward: .2e}")

import os

import cvxpy as cp
import numpy as np
import gurobipy
import pandas as pd

import gymnasium as gym
import gym_examples

def perfect_no_solar():
    # get the data
    data = "real"
    nhistory = 10
    start_index = 76*4*24
    T = n_steps = (90*4*24)-start_index
    solar_coloc = True
    env = gym.make(
        "gym_examples/BatteryEnv-v0", 
        nhistory=nhistory, 
        data=data, 
        mode="delay", 
        avoid_penalty=True,
        start_index=start_index,
        max_episode_steps=n_steps, # can change length here!"
        solar_coloc=solar_coloc,
    )
    c_lmp = np.zeros(T)
    s, _ = env.reset()
    c_lmp[0] = s[1]
    for i in range(1,T):
        s, _, _, _, _ = env.step(1)
        c_lmp[i] = s[1]
    rt_scale = 0.25
    power = 100
    capacity = 400
    
    # create model
    x_buy = cp.Variable(T, boolean=True)
    x_sell = cp.Variable(T, boolean=True)
    x_soc = cp.Variable(T+1)
    
    A_buy = np.zeros((T,T))
    A_sell = np.zeros((T,T))
    A_battery = np.zeros((T,T+1))
    # battery_next - battery - 50 * (x_buy - x_sell) = 0
    for i in range(T):
        A_buy[i,i] = -rt_scale*power
        A_sell[i,i] = rt_scale*power
        A_battery[i,i] = -1
        A_battery[i,i+1] = 1
            
    #A@x_h - A@r_h <= hydrogen_cap
    prob = cp.Problem(cp.Maximize(c_lmp.T@(x_sell-x_buy)),
                      [x_buy + x_sell <= 1, \
                       x_soc[0] == 0, # 0 initial SoC
                       x_soc >= 0,
                       x_soc <= capacity,
                       A_battery@x_soc + A_buy@x_buy + A_sell@x_sell == 0])
        
    # solve model
    prob.solve(solver=cp.GUROBI)
    
    action_arr = np.zeros(T)
    total_reward_arr = np.zeros(T)
    total_reward = 0
    cost_queue = np.zeros(int(capacity/(rt_scale*power)))
    ct = 0 
    for t in range(T):
        lmp = c_lmp[t]
        a = 1
        if x_buy.value[t] == 1:
            if ct == len(cost_queue):
                print(f"Exceeded battery capacity of {len(cost_queue)} charges!")
                exit(-1)
            cost_queue[ct] = lmp
            total_reward -= lmp * rt_scale * power
            ct += 1
            a = 0
        elif x_sell.value[t] == 1:
            oldest_cost = cost_queue[0]
            profit = (lmp - oldest_cost)*power
            cost_queue[1:ct] = cost_queue[:ct-1]
            ct -= 1
            # total_reward += profit
            total_reward += lmp * rt_scale * power
            a = 2
    
        action_arr[t] = a
        total_reward_arr[t] = total_reward
    
    # save
    data_arr = np.array([x_soc.value[1:], c_lmp, action_arr, total_reward_arr]).T
    fmt="%1.2f,%1.2f,%i,%1.2e"
    fname = os.path.join("logs", f"perfect_solar_coloc={solar_coloc}.csv")
    with open(fname, "wb") as fp:
        fp.write(b"soc,lmp,a,total_rwd\n")
        np.savetxt(fp, data_arr, fmt=fmt)
    print(f"Saved perfect data (total reward: {total_reward}")

def perfect_with_solar():
    # get the data
    data = "real"
    nhistory = 10
    start_index = 76*4*24
    T = n_steps = (90*4*24)-start_index
    env = gym.make(
        "gym_examples/BatteryEnv-v0", 
        nhistory=nhistory, 
        data=data, 
        mode="delay", 
        avoid_penalty=True,
        start_index=start_index,
        max_episode_steps=n_steps, # can change length here!"
        solar_coloc=True,
    )
    c_lmp = np.zeros(T)
    solar = np.zeros(T)
    s, _ = env.reset()
    c_lmp[0] = s[1]
    solar[0] = s[1+nhistory]
    for i in range(1,T):
        s, _, _, _, _ = env.step(1)
        c_lmp[i] = s[1]
        solar[i] = s[1+nhistory]
    rt_scale = 0.1
    power = 100
    capacity = 400
    
    # create model
    x_buy = cp.Variable(T, boolean=True)
    x_sell = cp.Variable(T, boolean=True)
    x_soc = cp.Variable(T+1)
    
    A_buy = np.zeros((T,T))
    A_sell = np.zeros((T,T))
    A_battery = np.zeros((T,T+1))
    # battery_next - battery - 50 * (x_buy - x_sell) = 0
    for i in range(T):
        A_buy[i,i] = -rt_scale*power
        A_sell[i,i] = rt_scale*power
        A_battery[i,i] = -1
        A_battery[i,i+1] = 1
            
    #A@x_h - A@r_h <= hydrogen_cap
    prob = cp.Problem(cp.Maximize(c_lmp.T@(x_sell-x_buy)),
                      [x_buy + x_sell <= 1, \
                       x_soc[0] == 0, # 0 initial SoC
                       x_soc >= 0,
                       x_soc <= capacity,
                       A_battery@x_soc + A_buy@x_buy + A_sell@x_sell == 0])
        
    # solve model
    prob.solve(solver=cp.GUROBI)
    
    action_arr = np.zeros(T)
    total_reward_arr = np.zeros(T)
    total_reward = 0
    cost_queue = np.zeros(int(capacity/(rt_scale*power)))
    ct = 0 
    for t in range(T):
        lmp = c_lmp[t]
        a = 1
        if x_buy.value[t] == 1:
            if ct == len(cost_queue):
                print(f"Exceeded battery capacity of {len(cost_queue)} charges!")
                exit(-1)
            cost_queue[ct] = lmp
            total_reward -= lmp * rt_scale * power
            ct += 1
            a = 0
        elif x_sell.value[t] == 1:
            oldest_cost = cost_queue[0]
            profit = (lmp - oldest_cost)*power
            cost_queue[1:ct] = cost_queue[:ct-1]
            ct -= 1
            # total_reward += profit
            total_reward += lmp * rt_scale * power
            a = 2
    
        action_arr[t] = a
        total_reward_arr[t] = total_reward
    
    # save
    data_arr = np.array([x_soc.value[1:], c_lmp, action_arr, total_reward_arr]).T
    fmt="%1.2f,%1.2f,%i,%1.2e"
    fname = os.path.join("logs", f"perfect_solar_coloc={solar_coloc}.csv")
    with open(fname, "wb") as fp:
        fp.write(b"soc,lmp,a,total_rwd\n")
        np.savetxt(fp, data_arr, fmt=fmt)
    print(f"Saved perfect data (total reward: {total_reward}")

if __name__ == "__main__":
    perfect_no_solar()
    # kperfect_with_solar()

import os
import time

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

def perfect_wo_solar(pnode_id, test_season, test_start_date, test_len_dates, solar_scale):
    # get the data
    get_index = lambda x : 4*24*x
    T = get_index(test_len_dates)
    seed = 0
    power = 100
    capacity = 400
    eta = 0.93      # efficiency
    rt_scale = 0.25
    env = gym.make(
        id="gym_examples/BatteryEnv-v0", 
        battery_capacity=capacity,
        battery_power=power,
        pnode_id=pnode_id,
        nhistory=4, 
        season=test_season,
        index_offset=get_index(test_start_date),
        max_episode_steps=get_index(test_len_dates),
        mode='default',
        efficiency=eta,
        daily_cost=0,
        more_data=False,
        delay_cost=False,
        solar_coloc=True,
        solar_scale=solar_scale,
        seed=seed,
        reset_mode='zero',
        reset_offset=0,
    )

    c_lmp = np.zeros(T)
    solar = np.zeros(T)
    s, _ = env.reset()
    c_lmp[0] = s['lmps'][0]
    solar[0] = s['solars'][0]
    for i in range(1,T):
        s, _, _, _, _ = env.step(1)
        c_lmp[i] = s['lmps'][0]
        solar[i] = s['solars'][0]
    
    # create model
    x_buy = cp.Variable(T, boolean=True)
    x_sell = cp.Variable(T, boolean=True)
    x_soc = cp.Variable(T+1)
    x_rwd = cp.Variable(T)
    
    A_buy = np.zeros((T,T))
    A_sell = np.zeros((T,T))
    A_battery = np.zeros((T,T+1))
    # battery_next - battery - alpha*dt * (x_buy - x_sell) = 0
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
        
    # solve modee
    prob.solve(solver=cp.GUROBI)
    
    # get actions
    # 0:= sell, 1:= null, 2:= buy
    action_arr = np.zeros(T, dtype=int)
    for t in range(T):
        action_arr[t] = round(-1*x_sell.value[t] + 1*x_buy.value[t] + 1)

    class Agent():
        def __init__(self):
            self.ct = 0

        def act(self):
            if self.ct >= len(action_arr):
                print("received too many actions")
            a_t = action_arr[self.ct % len(action_arr)]
            self.ct += 1
            return a_t

    agent = Agent()
    get_action = lambda obs: agent.act()
    return env, get_action

def perfect_realistic(pnode_id, test_season, test_start_date, test_len_dates, solar_scale):
    """ 
    Mixed-integer linear program for battery with efficiency, degradation, and
    solar.
    """
    get_index = lambda x : int(4*24*x)
    T = get_index(test_len_dates)
    seed = 0
    power = 100
    capacity = 400
    eta = 1      # efficiency
    env = gym.make(
        id="gym_examples/BatteryEnv-v0", 
        battery_capacity=capacity,
        battery_power=power,
        pnode_id=pnode_id,
        nhistory=4, 
        season=test_season,
        index_offset=get_index(test_start_date),
        max_episode_steps=get_index(test_len_dates),
        mode='default',
        efficiency=eta,
        daily_cost=0,
        more_data=False,
        delay_cost=False,
        solar_coloc=True,
        solar_scale=solar_scale,
        seed=seed,
        reset_mode='zero',
        reset_offset=0,
    )

    c_lmp = np.zeros(T)
    solar = np.zeros(T)
    s, _ = env.reset()
    c_lmp[0] = s['lmps'][0]
    solar[0] = s['solars'][0]
    for i in range(1,T):
        # null action
        s, _, _, _, _ = env.step(1)
        c_lmp[i] = s['lmps'][0]
        solar[i] = s['solars'][0]

    rt_scale = 0.25 # 15 min intervals
    
    # binary
    z_buy  = cp.Variable(T, boolean=True)
    z_sell = cp.Variable(T, boolean=True)
    z_min  = cp.Variable(T, boolean=True)
    z_max  = cp.Variable(T, boolean=True)
    # z_full = cp.Variable(T, boolean=True)

    # state of charge and min/max value holders
    x_soc = cp.Variable(T+1)
    x_soc_2 = cp.Variable(T) # for degredation
    x_soc_3 = cp.Variable(T) # for transition
    x_soc_4 = cp.Variable(T) # for min
    # reward
    r = cp.Variable(T)

    # zero initial SoC
    constraints = [
        x_soc[0] == 0, 
        z_buy + z_sell <= 1,
        # x_soc_2[0] == 0,
        # x_soc_3[0] == 0,
        # z_buy >= 0,
        # z_buy <= 1,
        # z_sell>= 0,
        # z_sell<= 1,
        # z_min >= 0,
        # z_min <= 1,
        # z_max >= 0,
        # z_max <= 1,
    ]
    # big-M constants
    M = capacity
    beta_pct = 1-0.1 # threshold for degradation
    beta = 1-0.1/(4*24) # degradation
    M_2 = (power * rt_scale + np.max(np.abs(solar))) * np.max(np.abs(c_lmp))

    # battery degredation: https://stackoverflow.com/questions/55899166/build-milp-constraint-from-if-else-statements
    for t in range(T):
        """
        constraints += [
             beta_pct*capacity - x_soc[t] <= M*(1-z_full[t]),
            -beta_pct*capacity + x_soc[t] <= M*z_full[t]
        ]
        constraints += [
            x_soc_2[t] >= beta*x_soc[t] - M*(1-z_full[t]),
            x_soc_2[t] <= beta*x_soc[t] + M*(1-z_full[t]),
        ]
        constraints += [
            x_soc_2[t] >= x_soc[t] - M*z_full[t],
            x_soc_2[t] <= x_soc[t] + M*z_full[t],
        ]
        constraints += [
            x_soc_2[t] >= x_soc[t] - M*z_full[t],
            x_soc_2[t] <= x_soc[t] + M*z_full[t],
        ]
        """
        constraints += [x_soc_2[t] == x_soc[t]]

    eta = 1 # to make it more solvable
    # transition constraints
    for t in range(T):
        constraints += [
            x_soc_3[t] == x_soc_2[t] 
                            + eta*power*rt_scale*z_buy[t] 
                            - power*rt_scale*z_sell[t],
        ]

    # min: https://or.stackexchange.com/questions/1160/how-to-linearize-min-function-as-a-constraint
    for t in range(T):
        constraints += [x_soc_4[t] <= capacity]
        constraints += [x_soc_4[t] <= x_soc_3[t]]
        constraints += [
            x_soc_4[t] >= capacity - M*(1-z_min[t]),
            x_soc_4[t] >= x_soc_3[t] - M*z_min[t]
        ]

    # max: https://or.stackexchange.com/questions/711/how-to-formulate-linearize-a-maximum-function-in-a-constraint/712#712
    for t in range(T):
        constraints += [x_soc[t+1] >= 0]
        constraints += [x_soc[t+1] >= x_soc_4[t]]
        constraints += [
            x_soc[t+1] <= M*(1-z_max[t]),
            x_soc[t+1] <= x_soc_4[t] + M*z_max[t]
        ]

    # reward
    for t in range(T):
        constraints += [r[t] == c_lmp[t]*(x_soc_2[t] - x_soc[t+1] + solar[t])]
        """
        constraints += [
            r[t] >= c_lmp[t]*((1./eta)*(x_soc_2[t] - x_soc[t+1]) + solar[t]) - M_2*(1-z_buy[t]),
            r[t] <= c_lmp[t]*((1./eta)*(x_soc_2[t] - x_soc[t+1]) + solar[t]) + M_2*(1-z_buy[t]),
        ]
        constraints += [
            r[t] >= c_lmp[t]*(eta*(x_soc_2[t] - x_soc[t+1]) + solar[t]) - M_2*z_buy[t],
            r[t] <= c_lmp[t]*(eta*(x_soc_2[t] - x_soc[t+1]) + solar[t]) + M_2*z_buy[t],
        ]
        """
    objective = cp.Maximize(cp.sum(r))
    prob = cp.Problem(objective, constraints)
        
    print("solving MILP")
    s_time = time.time()
    prob.solve(solver=cp.GUROBI)
    print("finished solving MILP in %.1fs" % (time.time()-s_time))
    print("objective: %.4e" % (prob.objective.value))

    # get actions
    # 0:= sell, 1:= null, 2:= buy
    action_arr = np.zeros(T, dtype=int)
    for t in range(T):
        if np.sum(int(z_sell.value[t] > 1e-10) + int(z_buy.value[t] > 1e-10)) == 2:
            print("Not a fractional solution at time %i; z_sell_t=%.4e and z_buy_t=%.4e" % (t, z_sell.value[t], z_buy.value[t]))

        if z_sell.value[t] >= 0.9:
            action_arr[t] = 0
        elif z_buy.value[t] >= 0.9:
            action_arr[t] = 2
        else:
            action_arr[t] = 1

    class Agent():
        def __init__(self):
            self.ct = 0

        def act(self):
            if self.ct >= len(action_arr):
                print("received too many actions")
            a_t = action_arr[self.ct % len(action_arr)]
            self.ct += 1
            return a_t

    agent = Agent()
    get_action = lambda obs: agent.act()
    return env, get_action

def perfect_realistic_2(pnode_id, test_season, test_start_date, test_len_dates, solar_scale):
    """ 
    Mixed-integer linear program for battery with efficiency, degradation, and
    solar.
    """
    get_index = lambda x : 4*24*x
    T = get_index(test_len_dates)
    seed = 0
    power = 100
    capacity = 400
    eta = 0.93      # efficiency
    env = gym.make(
        id="gym_examples/BatteryEnv-v0", 
        battery_capacity=capacity,
        battery_power=power,
        pnode_id=pnode_id,
        nhistory=4, 
        season=test_season,
        index_offset=get_index(test_start_date),
        max_episode_steps=get_index(test_len_dates),
        mode='default',
        efficiency=eta,
        daily_cost=0,
        more_data=False,
        delay_cost=False,
        solar_coloc=True,
        solar_scale=solar_scale,
        seed=seed,
        reset_mode='zero',
        reset_offset=0,
    )

    c_lmp = np.zeros(T)
    solar = np.zeros(T)
    s, _ = env.reset()
    c_lmp[0] = s['lmps'][0]
    solar[0] = s['solars'][0]
    for i in range(1,T):
        # null action
        s, _, _, _, _ = env.step(1)
        c_lmp[i] = s['lmps'][0]
        solar[i] = s['solars'][0]

    rt_scale = 0.25 # 15 min intervals
    
    # binary
    # z_buy  = cp.Variable(T, boolean=True)
    z_buy  = cp.Variable(T)
    z_sell = cp.Variable(T)
    z_min  = cp.Variable(T)
    z_max  = cp.Variable(T)
    z_full = cp.Variable(T)

    # state of charge and min/max value holders
    x_soc = cp.Variable(T+1)
    x_tsoc = cp.Variable(T)
    p = cp.Variable(T)
    q = cp.Variable(T)

    # reward
    r = cp.Variable(T)

    constraints = [
        z_buy <= 1,
        z_buy >= 0,
        z_sell<= 1,
        z_sell>= 0,
        z_min <= 1,
        z_min >= 0,
        z_max <= 1,
        z_max >= 0,
        z_full<= 1,
        z_full>= 0,
    ]

    # big-M constants
    M = capacity
    M_2 = np.max(np.abs(c_lmp)) * (power*rt_scale + np.max(np.abs(solar)))
    beta_pct = 1-0.1 # threshold for degradation
    beta = 1-0.1/(4*24) # degradation

    # transition constraints
    for t in range(T):
        constraints += [
            x_soc[t+1] >= x_tsoc[t] + eta*p[t] - M*(1-z_buy[t]),
            x_soc[t+1] <= x_tsoc[t] + eta*p[t] + M*(1-z_buy[t])
        ]
        constraints += [
            x_soc[t+1] >= q[t] - M*(1-z_sell[t]),
            x_soc[t+1] <= q[t] + M*(1-z_sell[t])
        ]
        constraints += [
            x_soc[t+1] >= x_tsoc[t] - M*(z_buy[t] + z_sell[t]),
            x_soc[t+1] <= x_tsoc[t] + M*(z_buy[t] + z_sell[t])
        ]
        constraints += [z_buy[t] + z_sell[t] <= 1]

    # min: https://or.stackexchange.com/questions/1160/how-to-linearize-min-function-as-a-constraint
    for t in range(T):
        constraints += [p[t] <= capacity - x_tsoc[t]]
        constraints += [p[t] <= power*rt_scale]
        constraints += [
            p[t] >= capacity - x_tsoc[t] - M*(1-z_min[t]),
            p[t] >= power*rt_scale - M*z_min[t]
        ]

    # max: https://or.stackexchange.com/questions/711/how-to-formulate-linearize-a-maximum-function-in-a-constraint/712#712
    for t in range(T):
        constraints += [q[t] >= 0]
        constraints += [q[t] >= x_tsoc[t] - power*rt_scale]
        constraints += [
            q[t] <= M*(1-z_max[t]),
            q[t] <= x_tsoc[t] - power*rt_scale + M*z_max[t]
        ]

    # battery degredation: https://stackoverflow.com/questions/55899166/build-milp-constraint-from-if-else-statements
    for t in range(T):
        constraints += [
            x_tsoc[t] >= beta*x_soc[t] - M*(1-z_full[t]),
            x_tsoc[t] <= beta*x_soc[t] + M*(1-z_full[t])
        ]
        constraints += [
            x_tsoc[t] >= x_soc[t] - M*z_full[t],
            x_tsoc[t] <= x_soc[t] + M*z_full[t]
        ]
        constraints += [
            beta_pct*capacity - x_soc[t] <= M*(1-z_full[t]),
            -beta_pct*capacity + x_soc[t] <= M*z_full[t]
        ]

    # reward
    for t in range(T):
        constraints += [
            r[t] >= c_lmp[t]*((1./eta)*(x_tsoc[t]-x_soc[t+1]) + solar[t]) - M_2*(1-z_buy[t]),
            r[t] <= c_lmp[t]*((1./eta)*(x_tsoc[t]-x_soc[t+1]) + solar[t]) + M_2*(1-z_buy[t]),
        ]
        constraints += [
            r[t] >= c_lmp[t]*(eta*(x_tsoc[t]-x_soc[t+1]) + solar[t]) - M_2*z_buy[t],
            r[t] <= c_lmp[t]*(eta*(x_tsoc[t]-x_soc[t+1]) + solar[t]) + M_2*z_buy[t],
        ]

    objective = cp.Maximize(cp.sum(r))
    prob = cp.Problem(objective, constraints)
        
    # solve model
    print("solving MILP relaxation")
    s_time = time.time()
    prob.solve(solver=cp.GUROBI)
    print("finished solving MILP relaxation in %.1fs" % (time.time()-s_time))
    print("objective value: %.4e" % prob.objective.value)

    # get actions
    # 0:= sell, 1:= null, 2:= buy
    action_arr = np.zeros(T)
    for t in range(T):
        action_arr[t] = np.argmax([z_sell.value[t], 1-z_sell.value[t]-z_buy.value[t], z_buy.value[t]])

    class Agent():
        def __init__(self):
            self.ct = 0

        def act(self):
            if self.ct >= len(action_arr):
                print("received too many actions")
            a_t = action_arr[self.ct % len(action_arr)]
            self.ct += 1
            return a_t

    agent = Agent()
    get_action = lambda obs: agent.act()
    return env, get_action

if __name__ == "__main__":
    # perfect_no_solar()
    # kperfect_with_solar()
    # perfect_realistic('PAULSWT_1_N013', 'w23', solar_scale=0.25)
    pass

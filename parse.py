import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def parse_csv(fname):
    df = pd.read_csv(fname, delimiter=",", header="infer")
    soc_arr = df["soc"].to_numpy()
    lmp_arr = df["lmp"].to_numpy()
    rwd_arr = df["total_rwd"].to_numpy()

    return (soc_arr, lmp_arr, rwd_arr)

def plot_initial_dqn_results():
    """ We will make two plots. 
        1. Median and decile performance for total reward
        2. SOC vs LMP for all trials
    """
    seeds = 10

    # get initial file size
    fname = os.path.join("logs", f"dqn_env_mode=delay_train_len=100000_seed=0.csv")
    (soc_arr, _, _) = parse_csv(fname)
    n = len(soc_arr)

    soc_data_arr = np.zeros((seeds, n))
    lmp_data_arr = np.zeros((seeds, n))
    rwd_data_arr = np.zeros((seeds, n))
    
    for i, seed in enumerate(range(seeds)):
        fname = os.path.join("logs", f"dqn_env_mode=delay_train_len=100000_seed={seed}.csv")
        (soc_arr, lmp_arr, rwd_arr) = parse_csv(fname)

        soc_data_arr[i,:] = soc_arr
        lmp_data_arr[i,:] = lmp_arr
        rwd_data_arr[i,:] = rwd_arr

    sorted_rwd_data_arr = np.sort(rwd_data_arr, axis=0)

    xs = np.arange(n)
    # plt.style.use('ggplot')
    plt.style.use('fivethirtyeight')
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,8))
    
    # First plot
    i_10 = int(seeds/10)
    i_50 = int(seeds/2)-1
    i_90 = int(seeds*9/10)-1
    axes[0].plot(xs, sorted_rwd_data_arr[i_50], color="black")
    axes[0].fill_between(xs, sorted_rwd_data_arr[i_10],sorted_rwd_data_arr[i_90], color="black", alpha=0.15)
    axes[0].set(title="Cumulative reward for periodic (10-50-90 deciles)", ylabel="Reward ($)", xlabel="Time steps")

    # Second plot
    m = n # 192
    m = 96 * 5
    par1 = axes[1].twinx()

    for seed in range(1,seeds):
        axes[1].plot(xs[:m], soc_data_arr[seed][:m], color=(0,0,0,0.05*seed))
    p1, = axes[1].plot(xs[:m], soc_data_arr[0][:m], label="SOC", color=(0,0,0,0))

    p2, = par1.plot(xs[:m], lmp_data_arr[0][:m], color="green", label="LMP ($/MWh)")

    axes[1].yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())
    tkw = dict(size=4, width=1.5)
    axes[1].tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)

    axes[1].set(title="Runs' SOC (each run has own shade)", xlabel="Time step", ylabel="SOC")
    par1.set(ylabel="LMP")

    lines = [p1, p2]
    axes[1].legend(lines, [l.get_label() for l in lines], loc=3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_initial_dqn_results()

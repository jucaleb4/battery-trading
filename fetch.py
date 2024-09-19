"""
Simple code to fetch LMP prices and download it.
"""
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


class LSTM(nn.Module):

    def __init__(self,input_size = 1, hidden_size = 50, out_size = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size,out_size)
        self.hidden = (torch.zeros(1,1,hidden_size),torch.zeros(1,1,hidden_size))

    def forward(self,seq):
        lstm_out, self.hidden = self.lstm(seq.view(len(seq),1,-1), self.hidden)
        pred = self.linear(lstm_out.view(len(seq),-1))
        return pred[-1]

def fetch_pnode_lmp(pnode_id, season):
    """
    Gathers 90 days worth of RT LMPs for a pnode during a particular season
    (recall we use the first 90-7 for training and remaining 7 for testing) and
    saves it as a csv file.
    """
    import gymnasium as gym
    import gym_examples

    get_index = lambda x : 4*24*x
    start_date = 0
    len_date = 90
    env = gym.make(
        id="gym_examples/BatteryEnv-v0", 
        pnode_id=pnode_id,
        season=season,
        index_offset=get_index(start_date),
        max_episode_steps=get_index(len_date),
    )

    rt_lmp_arr = []
    s, info = env.reset()
    rt_lmp_arr.append(s['lmps'][0])

    for t in range(1, get_index(len_date)):
        s, _, _, _, info = env.step(1)
        rt_lmp_arr.append(s['lmps'][0])

    # save file
    folder_name = "lmps"
    if not os.path.exists(folder_name):
        os.path.mkdir(folder_name)
    fname = "%s/%s_%s_rt_lmps.csv" % (folder_name, pnode_id, season)
    fp = open(fname, "w+")
    for t in range(len(rt_lmp_arr)):
        fp.write("%.2f" % rt_lmp_arr[t])
        if t < len(rt_lmp_arr)-1:
            fp.write("\n")
    fp.close()

def input_data(seq,ws):
    """ Create overlapping sequences of data """
    out = []
    L = len(seq)
    
    for i in range(L-ws):
        window = seq[i:i+ws]
        label = seq[i+ws:i+ws+1]
        out.append((window,label))
    
    return out

def predict_prices(pnode_id, season, max_duration=3600, stepsize="adam"):
    """ 
    Trains LSTM model on first 84 days, then we predict remaining 7 days 

    :param max_duration: max training time
    """
    # get the data
    folder_name = "lmps"
    fname = "%s/%s_%s_rt_lmps.csv" % (folder_name, pnode_id, season)
    rt_lmp_arr = np.squeeze(pd.read_csv(fname).to_numpy())
    y = torch.from_numpy(rt_lmp_arr).to(torch.float32)
    x = torch.arange(len(rt_lmp_arr))

    test_size = 7*24*4   # 1 week
    train_set = y[:-test_size]
    test_set = y[-test_size:]
    window_size = 1*24*4 # 1 day
    train_data = input_data(train_set, window_size)

    # initalize LSTM model
    torch.manual_seed(42)
    model = LSTM(hidden_size=100)
    criterion = nn.MSELoss()
    if stepsize == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    epochs = 100
    future = 7*24*4 # amount to predict
    n_epochs = 0
    s_time = time.time()

    for i in range(epochs):
        n_epochs += 1

        for seq, y_train in train_data:
            optimizer.zero_grad()
            model.hidden = (torch.zeros(1,1,model.hidden_size),
                           torch.zeros(1,1,model.hidden_size))

            y_pred = model(seq)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()

        print(f'Epoch: {i+1:2} Loss: {loss.item():10.8f} (time: {time.time() - s_time:.0f}s)')

        # predict prices
        preds = train_set[-window_size:].tolist()
        for f in range(future):
            seq = torch.FloatTensor(preds[-window_size:])
            with torch.no_grad():
                model.hidden = (torch.zeros(1,1,model.hidden_size),
                               torch.zeros(1,1,model.hidden_size))
                preds.append(model(seq).item())

        preds = preds[window_size:]
        # save last set of predicted prices
        fname = "%s/%s_%s_predicted_rt_lmps_stepsize=%s_epoch=%d.csv" % (folder_name, pnode_id, season, stepsize, n_epochs)
        fp = open(fname, "w+")
        for t in range(len(preds)):
            fp.write("%.2f" % preds[t])
            if t < len(preds)-1:
                fp.write("\n")
        fp.close()

        # break after 1 hour
        if time.time() - s_time > max_duration:
            break

    print(f'\nDuration: {time.time() - s_time:.0f} seconds')

    # plot 
    fname = "%s/%s_%s_test_v_predicted_rt_lmps_stepsize=%s.png" % (folder_name, pnode_id, season, stepsize)
    plt.style.use("ggplot")
    ax = plt.subplot()
    ax.plot(test_set, label="true", linestyle="solid", color="black")
    ax.plot(preds, label="pred", linestyle="dashed", color="red")
    ax.set(title="Predicted vs actual RT LMP (%d epochs)" % n_epochs)
    plt.tight_layout()
    plt.savefig(fname, dpi=240)

seasons = ['S23', 'w23']
stepsizes = ["sgd", "adam"]
pnodes  = ['PAULSWT_1_N013', 'COTWDPGE_1_N001', 'ALAMT3G_7_B1']

for season in seasons:
    for stepsize in stepsizes:
        predict_prices(pnodes[1], seasons[0], max_duration=3600, stepsize=stepsize)

for pnode in pnodes:
    for season in seasons:
        fetch_pnode_lmp(pnode, season)

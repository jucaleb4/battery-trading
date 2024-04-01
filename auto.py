import os

# for i in range(1):
    # cmd = f"python run.py --env_mode delay --seed {i} --train_len 100000 --norm_obs"
    # os.system(cmd)
    # cmd = f"python run.py --env_mode delay --seed {i} --train_len 100000 --norm_obs --more_data"
    # os.system(cmd)
   #  cmd = f"python run.py --train_len 200000 --more_data --norm_obs --norm_rwd --daily_cost 100 --learning_rate 0.000918 --exploration_fraction 0.9685 --seed {i}"
    # os.system(cmd)

# cmd = f"python run.py --train_len 200000 --more_data --norm_obs --norm_rwd --daily_cost 100 --learning_rate 0.000918 --exploration_fraction 0.9685 --n_trials 10 --seed 0"
cmd = f"python run.py --settings_file settings/s1.json"
os.system(cmd)

cmd = f"python run.py --settings_file settings/s2.json"
os.system(cmd)

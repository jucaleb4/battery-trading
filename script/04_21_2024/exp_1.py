import os
import sys

import argparse
from collections import OrderedDict
import json

MAX_RUNS = 12
DATE = "04_21_2024"
EXP_ID  = 1

def parse_sub_runs(sub_runs):
    start_run_id, end_run_id = 0, MAX_RUNS
    if (sub_runs is not None):
        try:
            start_run_id, end_run_id = sub_runs.split(",")
            start_run_id = int(start_run_id)
            end_run_id = int(end_run_id)
            assert 0 <= start_run_id <= end_run_id <= MAX_RUNS, "sub_runs id must be in [0,%s]" % (MAX_RUNS-1)
            
        except:
            raise Exception("Invalid sub_runs id. Must be two integers split between [0,%s] split by a single comma with no space" % (MAX_RUNS-1))

    return start_run_id, end_run_id

def setup_setting_files(seed_0, max_trials, max_steps):
    od = OrderedDict([
        ("alg", "ppo"),
        ("max_trials", max_trials),
        ("pnode_id", "MIL1_3_PASGNODE"),
        ("seed", seed_0),
        ("n_history", 16),
        ("max_iters", 64), # this is for bangbang
        ("max_steps", max_steps), # this is for q-learning
        ("env_mode", "default"),
        ("norm_obs", False),
        ("norm_rwd", False),
        ("more_data", True),
        ("daily_cost", 0.0),
        ("delay_cost", False),
        ("solar_coloc", True),
        ("solar_scale", 0.0),
        ("solar_scale_test", 0.0),
        ("reset_mode", "rand"), 
        ("reset_offset", 2048),
        ("policy_type", "MlpPolicy"),
        ("n_steps", 2048),
        ("batch_size", 64),
        ("n_epochs", 10),
        ("gamma", 0.99),
        ("gae_lambda", 0.95),
        ("normalize_advantage", True),
        ("max_grad_norm", 0.5),
        # ("learning_starts", 20), # this and below are for QLearning
        # ("exploration_fraction", 0.9685),
        # ("exploration_final_eps", 0.05),
        # ("gradient_steps", -1),
        # ("target_update_interval", 540),
    ])
    log_folder_base = os.path.join("logs", DATE, "exp_%s" % EXP_ID)
    setting_folder_base = os.path.join("settings", DATE, "exp_%s" % EXP_ID)

    if not(os.path.exists(log_folder_base)):
        os.makedirs(log_folder_base)
    if not(os.path.exists(setting_folder_base)):
        os.makedirs(setting_folder_base)

    pnodes = ["MIL1_3_PASGNODE", "ALAMT3G_7_B1", "PAULSWT_1_GN002", "FREMNT_1_N013"]
    ct = 0

    # iterate over various pnodes
    for pnode_id in pnodes:
        od["pnode_id"] = pnode_id

        # create control with various penalties
        solars = [0.0, 0.25, 0.75]
        for solar in solars:
            setting_fname = os.path.join(setting_folder_base,  "run_%s.json" % ct)
            log_folder = os.path.join(log_folder_base, "run_%s" % ct)
            od["solar_scale"] = solar
            od["solar_scale_test"] = solar
            od["log_folder"] = log_folder
            if not(os.path.exists(od["log_folder"])):
                os.makedirs(od["log_folder"])
            with open(setting_fname, 'w', encoding='utf-8') as f:
                json.dump(od, f, ensure_ascii=False, indent=4)
            ct += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--setup", action="store_true", help="Setup environments. Otherwise we run the experiments")
    parser.add_argument("--run", action="store_true", help="Setup environments. Otherwise we run the experiments")
    parser.add_argument(
        "--mode", 
        type=str, 
        default="work", 
        choices=["full", "validate", "work"],
        help="Set up number of trials and max_step for various testing reasons"
    )
    parser.add_argument(
        "--sub_runs", 
        type=str, 
        help="Which experiments to run. Must be given as two integers separate by a comma with no space"
    )
    args = parser.parse_args()
    seed_0 = 0

    if args.setup:
        max_trials = 3
        max_steps = 200_000
        if args.mode == "validate":
            max_trials = 1
        elif args.mode == "work":
            max_trials = 1
            max_steps = 10_000

        setup_setting_files(seed_0, max_trials, max_steps)
    elif args.run:
        start_run_id, end_run_id = parse_sub_runs(args.sub_runs)
        folder_name = os.path.join("settings", DATE, "exp_%i" % EXP_ID)

        for i in range(start_run_id, end_run_id+1):
            settings_file = os.path.join(folder_name, "run_%i.json" % i)
            os.system("python run.py --settings %s" % settings_file)
    else:
        print("Neither setup nor run passed. Shutting down...")

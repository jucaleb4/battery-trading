import os
import sys

import argparse
from collections import OrderedDict
import json

MAX_RUNS = 3
DATE = "04_20_2024"
EXP_ID  = 0

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

def setup_setting_files(seed_0, max_trials):
    od = OrderedDict([
        ("alg", "bangbang"),
        ("max_trials", max_trials),
        ("pnode_id", "MIL1_3_PASGNODE"),
        ("seed", seed_0),
        ("n_history", 16),
        ("max_iters", 64),
        ("env_mode", "default"),
        ("norm_obs", True),
        ("norm_rwd", True),
        ("more_data", True),
        ("daily_cost", 0.0),
        ("delay_cost", True),
        ("solar_coloc", True),
        ("solar_scale", 0.0),
        ("solar_scale_test", 0.0),
        # ("policy_type", "MlpPolicy"),
        # ("learning_rate", 0.000918),
        # ("max_grad_norm", 10),
        # ("solar_scale_test", 0.25),
        # ("learning_starts", 20),
        # ("exploration_fraction", 0.9685),
        # ("exploration_final_eps", 0.05),
        # ("gradient_steps", -1),
        # ("batch_size", 48),
        # ("target_update_interval", 540),
    ])
    log_folder_base = os.path.join("logs", DATE, "exp_%s" % EXP_ID)
    setting_folder_base = os.path.join("settings", DATE, "exp_%s" % EXP_ID)

    if not(os.path.exists(log_folder_base)):
        os.makedirs(log_folder_base)
    if not(os.path.exists(setting_folder_base)):
        os.makedirs(setting_folder_base)

    ct = 0
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
        "--sub_runs", 
        type=str, 
        help="Which experiments to run. Must be given as two integers separate by a comma with no space"
    )
    args = parser.parse_args()
    seed_0 = 0

    if args.setup:
        max_trials = 3
        setup_setting_files(seed_0, max_trials)
    elif args.run:
        start_run_id, end_run_id = parse_sub_runs(args.sub_runs)
        folder_name = os.path.join("settings", DATE, "exp_%s" % EXP_ID)

        for i in range(start_run_id, end_run_id+1):
            settings_file = os.path.join(folder_name, "run_%s.json" % i)
            os.system("python run.py --settings %s" % settings_file)
    else:
        print("Neither setup nor run passed. Shutting down...")

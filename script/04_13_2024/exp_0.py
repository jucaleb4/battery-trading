import os
import sys

import argparse
from collections import OrderedDict
import json

max_runs = 19

def parse_sub_runs(sub_runs):
    start_run_id, end_run_id = 0, max_runs
    if (sub_runs is not None):
        try:
            start_run_id, end_run_id = sub_runs.split(",")
            start_run_id = int(start_run_id)
            end_run_id = int(end_run_id)
            assert 0 <= start_run_id <= end_run_id <= max_runs, "sub_runs id must be in [0,%s]" % max_runs
            
        except:
            raise Exception("Invalid sub_runs id. Must be two integers split between [0,%s] split by a single comma with no space" % max_runs)

    return start_run_id, end_run_id

def setup_setting_files(seed_0, max_trials, max_steps):
    od = OrderedDict([
        ("max_trials", max_trials),
        ("seed", seed_0),
        ("n_history", 16),
        ("max_steps", max_steps),
        ("env_mode", "default"),
        ("norm_obs", True),
        ("norm_rwd", True),
        ("more_data", True),
        ("daily_cost", 0.0),
        ("delay_cost", True),
        ("solar_coloc", True),
        ("solar_scale", 0.0),
        ("solar_scale_test", 0.0),
        ("policy_type", "MlpPolicy"),
        ("learning_rate", 0.000918),
        ("max_grad_norm", 10),
        ("solar_scale_test", 0.25),
        ("learning_starts", 20),
        ("exploration_fraction", 0.9685),
        ("exploration_final_eps", 0.05),
        ("gradient_steps", -1),
        ("batch_size", 48),
        ("target_update_interval", 540),
    ])

    folder_name = os.path.join("settings", "04_13_2024", "exp_0")
    if not(os.path.exists(folder_name)):
        os.makedirs(folder_name)
    for i in range(0,max_runs+1):
        log_folder_base = os.path.join("logs", "04_13_2024", "exp_0")
        if not(os.path.exists(log_folder_base)):
            os.makedirs(log_folder_base)

    ct = 0
    # create control with various penalties
    penalties = [0.0, 10.61, 106.10]
    solars = [0.0, 0.25, 0.75]
    for penalty in penalties:
        od["daily_cost"] = penalty
        for solar in solars:
            fname = os.path.join(folder_name, "run_%s.json" % ct)
            od["solar_scale"] = solar
            od["solar_scale_test"] = solar
            od["log_folder"] = os.path.join(log_folder_base,  "run_%s" % ct)
            if not(os.path.exists(od["log_folder"])):
                os.makedirs(od["log_folder"])
            with open(fname, 'w', encoding='utf-8') as f:
                json.dump(od, f, ensure_ascii=False, indent=4)
            ct += 1

    # reset
    od["daily_cost"] = 0.0

    solar_tests = [0.25, 0.75]
    # non-solar training
    for solar_test in solar_tests:
        fname = os.path.join(folder_name, "run_%s.json" % ct)
        od["solar_scale"] = 0.0
        od["solar_scale_test"] = solar_test
        od["log_folder"] = os.path.join(log_folder_base,  "run_%s" % ct)
        if not(os.path.exists(od["log_folder"])):
            os.makedirs(od["log_folder"])
        with open(fname, 'w', encoding='utf-8') as f:
            json.dump(od, f, ensure_ascii=False, indent=4)
        ct += 1

    # obs and reward normalization
    od["norm_obs"] = False
    od["norm_rwd"] = False
    for solar in solars:
        fname = os.path.join(folder_name, "run_%s.json" % ct)
        od["solar_scale"] = solar
        od["solar_scale_test"] = solar
        od["log_folder"] = os.path.join(log_folder_base,  "run_%s" % ct)
        if not(os.path.exists(od["log_folder"])):
            os.makedirs(od["log_folder"])
        with open(fname, 'w', encoding='utf-8') as f:
            json.dump(od, f, ensure_ascii=False, indent=4)
        ct += 1

    # reset
    od["norm_obs"] = True
    od["norm_rwd"] = True

    # non-contextual
    od["more_data"] = False
    for solar in solars:
        fname = os.path.join(folder_name, "run_%s.json" % ct)
        od["solar_scale"] = solar
        od["solar_scale_test"] = solar
        od["log_folder"] = os.path.join(log_folder_base,  "run_%s" % ct)
        if not(os.path.exists(od["log_folder"])):
            os.makedirs(od["log_folder"])
        with open(fname, 'w', encoding='utf-8') as f:
            json.dump(od, f, ensure_ascii=False, indent=4)
        ct += 1

    # reset
    od["more_data"] = True

    # no reward shaping
    od["delay_cost"] = False
    for solar in solars:
        fname = os.path.join(folder_name, "run_%s.json" % ct)
        od["solar_scale"] = solar
        od["solar_scale_test"] = solar
        od["log_folder"] = os.path.join(log_folder_base,  "run_%s" % ct)
        if not(os.path.exists(od["log_folder"])):
            os.makedirs(od["log_folder"])
        with open(fname, 'w', encoding='utf-8') as f:
            json.dump(od, f, ensure_ascii=False, indent=4)
        ct += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--setup", action="store_true", help="Setup environments. Otherwise we run the experiments")
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
    else:
        start_run_id, end_run_id = parse_sub_runs(args.sub_runs)
        folder_name = os.path.join("settings", "04_13_2024", "exp_0")

        for i in range(start_run_id, end_run_id+1):
            settings_file = os.path.join(folder_name, "run_%s.json" % i)
            # with open(args.settings, "r") as fp:
            #     settings = json.load(fp)
            # run(settings)
            os.system("python run.py --settings %s" % settings_file)

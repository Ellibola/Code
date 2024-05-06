# Sytem utils
import argparse
import os
import datetime
import subprocess
import json
import shutil
# Pytorch related
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt

# Constants
ROOT_TO_JSON = "./Configurations"
ROOT_TO_HISTORY = "./History"
# Parse experiment indexes
parser = argparse.ArgumentParser(description='Top training framework')
parser.add_argument(
    '-idx', '--exp_idx',
    type=int,
    nargs='+',
    help='list containing experiment indexes',
    default=[0],
    required=False
)
args = parser.parse_args()

def main():
    process_list = []
    start_time = datetime.datetime.now().strftime('%Y_%m%d_%H_%M_%S')
    # Read dict from JSON file and start according thread
    for idx in args.exp_idx:
        root_json = ROOT_TO_JSON + os.sep + "Config_" + str(idx) + ".json"
        with open(root_json, 'r') as f:
            config_dict = json.load(f)
        # Create a history folder
        root_folder = ROOT_TO_HISTORY + os.sep + start_time + "_EXP_" + str(idx)
        if not os.path.exists(root_folder):
            os.mkdir(root_folder)
        # Save the setting file
        shutil.copyfile(root_json, root_folder+os.sep+"settings.json")
        # Launch subprocess according to training type
        if config_dict["TRAINING_TYPE"]=='ol':
            process_list.append(
                subprocess.Popen(['python', '-u', './online_training.py', '-idx', str(idx), '-root', root_folder])
                )
        elif config_dict["TRAINING_TYPE"]=='normal':
            process_list.append(
                subprocess.Popen(['python', '-u', './normal_training.py', '-idx', str(idx), '-root', root_folder])
                )
        elif config_dict["TRAINING_TYPE"]=='ol_ftrl':
            process_list.append(
                subprocess.Popen(['python', '-u', './normal_training.py', '-idx', str(idx), '-root', root_folder])
                )
        else:
            raise NotImplementedError
if __name__ == "__main__":
    main()
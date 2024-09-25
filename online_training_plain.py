# Sytem utils
import argparse
import os
import datetime
import subprocess
import json
import shutil
import sys
# Pytorch related
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
# Models
from models.model_wizard import model_wizard
# Dataset
from data.data_wizard import data_wizard
# Utils
from utils import *
import csv
# Customized optimizers
from optimizers.ftrl import FTRL
from optimizers.obc import OBC
# Optuna
import optuna

""" Parse experiment indexes """
parser = argparse.ArgumentParser(description='T-model training framework')
parser.add_argument(
    '-idx', '--exp_idx',
    type=int,
    help='experiment index',
    default=0,
    required=True
)
parser.add_argument(
    '-root', '--root',
    type=str,
    help='root to the folder stores the history',
    default="./",
    required=True
)
args = parser.parse_args()

""" Redirect stdout to file"""
orig_stdout = sys.stdout
f = open(args.root + os.sep + "log.out", 'w')
sys.stdout = f
""" Redirect stderr to file"""
orig_stderr = sys.stderr
f_err = open(args.root + os.sep + "log_err.out", 'w')
sys.stderr = f_err

""" Read configuration from file """
with open(args.root + os.sep + "settings.json", 'r') as f:
    config = json.load(f)

""" The device runing the model """
# For Apple silicon
DEVICE = torch.device("mps")
# For Linux Server
if (config['GPU_IDX'] if 'GPU_IDX' in config.keys() else False):
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


""" Create model, optimizer, criterion"""
def get_pytorch_obj():
    model = model_wizard(
        dataset=config['DATASET'],
        bit_w=config['W_BIT'] if 'W_BIT' in config.keys() else 32,
        bit_a=config['A_BIT'] if 'A_BIT' in config.keys() else 32,
        version=config['MODEL_VER'],
        device=DEVICE,
        if_avg=config['IF_PARAM_AVG'] if 'IF_PARAM_AVG' in config.keys() else False,
        gamma=config['GAMMA'] if 'GAMMA' in config.keys() else 0.99
    ).to(DEVICE)
    # Set the hyper parameter of the online CNN
    model.set_hyper_params(
        beta=config['BETA'],
        s=config['SMOOTH_FACTOR']
    )
    # Setting up parameter groups and weight decay
    weight_decay = config['WEIGHT_DECAY']
    param_group = get_para_group(model, weight_decay) if weight_decay!=0 else model.parameters()
    # If we need to use OBC or not
    lr = config['LR']
    if (config['IF_OBC'] if 'IF_OBC' in config.keys() else False):
        optimizer = \
        OBC(
            params=param_group,
            base_optimizer_class=opt.AdamW,
            eta=config['OBC_ETA'] if 'OBC_ETA' in config.keys() else 0.99,
            lr=lr
        ) \
        if config['OPT']=="adamw" else \
        OBC(
            params=param_group,
            base_optimizer_class=FTRL,
            eta=config['OBC_ETA'] if 'OBC_ETA' in config.keys() else 0.99,
            alpha=config['ALPHA']
        ) \
        if config['OPT']=="ftrl" else \
        OBC(
            params=param_group,
            base_optimizer_class=opt.SGD,
            eta=config['OBC_ETA'] if 'OBC_ETA' in config.keys() else 0.99,
            lr=lr,
            momentum=0.9
        )
    else:
        optimizer = \
        opt.AdamW(
            params=param_group,
            lr=lr
        ) \
        if config['OPT']=="adamw" else \
        FTRL(
            params=param_group,
            alpha=config['ALPHA']
        ) \
        if config['OPT']=="ftrl" else \
        opt.SGD(
            params=param_group,
            lr=lr,
            momentum=0.9
        )
    # Learning rate scheduler
    train_loader, _, _ = get_dataset()
    itr_total = config['NUM_EPOCH'] * len(train_loader)
    lr_sch = GradualWarmupScheduler(
        optimizer=optimizer, 
        max_iter=itr_total, 
        min_lr=0, 
        base_lr=config['LR'], 
        warmup_lr=50*config['LR'], 
        warmup_steps=0.01*itr_total
    ) if (config['IF_LR_SCH'] if 'IF_LR_SCH' in config.keys() else False) else None

    return model, optimizer, lr_sch

""" Dataset """
def get_dataset():
    train_loader, val_loader, test_loader = data_wizard(
        name=config['DATASET'],
        batch_size=config['BATCH_SIZE'],
        val_par=config['VAL_PAR'] if 'VAL_PAR' in config.keys() else 0.1,
    )
    return train_loader, val_loader, test_loader


""" Save to file """
def save2file(model):
    # Save the trained model
    torch.save(model.state_dict(), args.root+os.sep+'online_model')

""" Objective, the training pipeline """
def objective():
    # Get pytorch objs
    model, optimizer, lr_sch = get_pytorch_obj()
    # Get datasets
    train_loader, _, val_loader = get_dataset()
    for epo in range(config['NUM_EPOCH']):
        # Training
        for images, labels in train_loader:
            # Classic pytorch pipeline
            with torch.autograd.set_detect_anomaly(False):
                model.step(images.to(DEVICE), labels.to(DEVICE), optimizer)
            if lr_sch is not None:
                lr_sch.step()
        # Validate
        acc_t1, _ = validation_t1_t5(model, val_loader, DEVICE)
        print("The accuracy:{}% @ {} epoch.\n".format(acc_t1, epo))
    return model

""" Main function """
def main():
    model = objective()
    
    # Test model accuracy on the test set after training
    _, _, test_loader = get_dataset()
    acc_t1, acc_t5 = validation_t1_t5(model, test_loader, DEVICE)
    print("Testset acc_t1:{:.3f}% with {} parameters.".format(acc_t1, para_count(model)))
    print("Testset acc_t5:{:.3f}% with {} parameters.\n".format(acc_t5, para_count(model)))
    # Save
    save2file(model)


if __name__ == "__main__":
    main()

""" Redirect stdout to file"""
sys.stdout = orig_stdout
f.close()

""" Redirect stderr back"""
sys.stderr = orig_stderr
f_err.close()
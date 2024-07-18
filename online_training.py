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
        gamma=config['GAMMA'] if 'GAMMA' in config.keys() else 0.99,
    ).to(DEVICE)
    # Set the hyper parameter of the online CNN
    model.set_hyper_params(
        beta=config['BETA'],
        s=config['SMOOTH_FACTOR']
    )
    # Setting up parameter groups and weight decay
    weight_decay = config['WEIGHT_DECAY'] if 'WEIGHT_DECAY' in config.keys() else 0.0
    param_group = get_para_group(model, weight_decay) if weight_decay!=0 else model.parameters()
    # If we need to use OBC or not
    if (config['IF_OBC'] if 'IF_OBC' in config.keys() else False):
        optimizer = \
        OBC(
            params=param_group,
            base_optimizer_class=opt.AdamW,
            eta=config['OBC_ETA'] if 'OBC_ETA' in config.keys() else 0.99,
            lr=config['LR']
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
            lr=config['LR'],
            momentum=0.9
        )
    else:
        optimizer = \
        opt.AdamW(
            params=param_group,
            lr=config['LR']
        ) \
        if config['OPT']=="adamw" else \
        FTRL(
            params=param_group,
            alpha=config['ALPHA']
        ) \
        if config['OPT']=="ftrl" else \
        opt.SGD(
            params=param_group,
            lr=config['LR'],
            momentum=0.9
        )
    return model, optimizer

""" Dataset """
def get_dataset():
    train_loader, val_loader, test_loader = data_wizard(
        name=config['DATASET'],
        batch_size=config['BATCH_SIZE'],
        val_par=config['VAL_PAR'] if 'VAL_PAR' in config.keys() else 0.1,
    )
    return train_loader, val_loader, test_loader


""" Save to file """
def save2file(model, if_val, val_acc, loss_list):
    # Save the result
    if if_val is not None:
        with open(args.root+os.sep+'val_acc.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['top1', 'top5'])
            writer.writerows(val_acc)

    with open(args.root+os.sep+'loss.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['loss', 'lr'])
            writer.writerows(loss_list)

    # Save the trained model
    torch.save(model.state_dict(), args.root+os.sep+'online_model')

""" Main function """
def main():
    validated_acc = []
    loss_list = []
    # Get pytorch objs
    model, optimizer = get_pytorch_obj()
    print("The model to be trained is:{}".format(model.__class__.__name__))
    # Load the saved state:
    if ("CONTINUE_FROM_SAVED" in config.keys()) and \
       ("ROOT_TO_SAVED" in config.keys()) and config["CONTINUE_FROM_SAVED"]:
        model.load_state_dict(torch.load(config["ROOT_TO_SAVED"], map_location=DEVICE))
    # Get datasets
    train_loader, test_loader, val_loader = get_dataset()
    # Learning rate scheduler
    itr_total = config['NUM_EPOCH'] * len(train_loader)
    lr_sch = GradualWarmupScheduler(
        optimizer=optimizer, 
        max_iter=itr_total, 
        min_lr=0, 
        base_lr=config['LR'], 
        warmup_lr=50*config['LR'], 
        warmup_steps=0.01*itr_total
    ) if (config['IF_LR_SCH'] if 'IF_LR_SCH' in config.keys() else False) else None
    # Main training loop
    print("Training start, the model has {} parameters.\n\n".format(para_count(model)))
    # Total step count
    total_count = 0
    c1_sum = 0
    c5_sum = 0
    n_sum = 0
    for epo in range(config['NUM_EPOCH']):
        # Report process
        print('Process: {:.3f}%, with lr: {}'.format(
            epo / config['NUM_EPOCH'] * 100, 
            optimizer.param_groups[0]['lr'] if not config['OPT']=="ftrl" else None
        ))
        # Test the model acc on validation set(if applicable)
        if (epo % 1 == 0) and (val_loader is not None):
            acc_t1, acc_t5 = validation_t1_t5(model, val_loader, DEVICE)
            validated_acc.append([str(acc_t1), str(acc_t5)])
            print("Validation acc_t1:{:.3f}%".format(acc_t1))
            print("Validation acc_t5:{:.3f}% \n".format(acc_t5))
        # Variables for lr scheduler
        count = 0
        loss_accumulated = 0
        # Training
        for images, labels in train_loader:
            count += 1
            total_count += 1
            # Classic pytorch pipeline
            with torch.autograd.set_detect_anomaly(False):
                loss, y = model.step(images.to(DEVICE), labels.to(DEVICE), optimizer)
            # Append loss
            loss_accumulated += loss.item()
            loss_list.append([
                str(loss.item()), 
                str(optimizer.param_groups[0]['lr']) if not config['OPT']=="ftrl" else None
            ])
            # Predicted accuracy per 20 steps
            c1, c5, n = correct_count(y, labels.to(DEVICE))
            c1_sum += c1
            c5_sum += c5
            n_sum += n
            if (total_count % int(4000/config['BATCH_SIZE']) == 0)&(not (config['IF_MIXUP_CUTMIX'] if 'IF_MIXUP_CUTMIX' in config.keys() else False)):
                print('Current top1 acc: {:.3f}%'.format(c1_sum / n_sum * 100))
                print('Current top5 acc: {:.3f}% \n'.format(c5_sum / n_sum * 100))
                c1_sum = 0
                c5_sum = 0
                n_sum = 0
                # Save the check_point
                torch.save(model.state_dict(), args.root+os.sep+'model_checkpoint')
            # Constrain weights:
            for layer in model.modules():
                if (type(layer).__name__ == 'BConv2d') | \
                   (type(layer).__name__ == 'BLinear') | \
                   (type(layer).__name__ == 'BConv2d_first'):
                    layer.weight.data.copy_(torch.clamp(layer.weight.data.detach(), -1, 1))
            if lr_sch is not None:
                lr_sch.step()
    # Test model accuracy on the test set after training
    acc_t1, acc_t5 = validation_t1_t5(model, test_loader, DEVICE)
    validated_acc.append([str(acc_t1), str(acc_t5)])
    print("Testset acc_t1:{:.3f}% with {} parameters.".format(acc_t1, para_count(model)))
    print("Testset acc_t5:{:.3f}% with {} parameters.\n".format(acc_t5, para_count(model)))
    # Save
    save2file(model, config['IF_VAL'], validated_acc, loss_list)


if __name__ == "__main__":
    main()

""" Redirect stdout to file"""
sys.stdout = orig_stdout
f.close()
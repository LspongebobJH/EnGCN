import gc
import json
import os
import random
from datetime import datetime

import numpy as np
import torch

from options.base_options import BaseOptions
from trainer import trainer
from utils import print_args

from argparse import Namespace
from ray import tune
import wandb


def set_seed(args):
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    if args.cuda and not torch.cuda.is_available():  # cuda is not available
        args.cuda = False
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_num)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)


def pipe(config:dict):
    exp = config['exp']
    name = config['init']
    wandb.init(project=f"exp{exp}", config=config, dir='/mnt/jiahanli/wandb', name=name)
    args = Namespace(**config)

    # list_test_acc = []
    # list_valid_acc = []
    # list_train_acc = []

    # filedir = f"./logs/{args.dataset}"
    # if not os.path.exists(filedir):
    #     os.makedirs(filedir)
    # if not args.exp_name:
    #     filename = f"{args.type_model}.json"
    # else:
    #     filename = f"{args.exp_name}.json"
    # path_json = os.path.join(filedir, filename)

    # try:
    #     resume_seed = 0
    #     if os.path.exists(path_json):
    #         if args.resume:
    #             with open(path_json, "r") as f:
    #                 saved = json.load(f)
    #                 resume_seed = saved["seed"] + 1
    #                 list_test_acc = saved["test_acc"]
    #                 list_valid_acc = saved["val_acc"]
    #                 list_train_loss = saved["train_loss"]
    #         else:
    #             t = os.path.getmtime(path_json)
    #             tstr = datetime.fromtimestamp(t).strftime("%Y_%m_%d_%H_%M_%S")
    #             os.rename(
    #                 path_json, os.path.join(filedir, filename + "_" + tstr + ".json")
    #             )
    #     if resume_seed >= args.N_exp:
    #         print("Training already finished!")
    #         return
    # except:
    #     pass

    print_args(args)

    if args.debug_mem_speed:
        trnr = trainer(args)
        trnr.mem_speed_bench()

    # for seed in range(resume_seed, args.N_exp):
    # for seed in range(0, args.N_exp):
    # print(f"seed (which_run) = <{seed}>")

    set_seed(args)
    # torch.cuda.empty_cache()
    trnr = trainer(args)
    if args.type_model in [
        "SAdaGCN",
        "AdaGCN",
        "GBGCN",
        "AdaGCN_CandS",
        "AdaGCN_SLE",
        "EnGCN",
    ]:
        train_acc, valid_acc, test_acc = trnr.train_ensembling(args.random_seed)
    else:
        train_acc, valid_acc, test_acc = trnr.train_and_test(args.random_seed)
    wandb.log({'final_train_acc': train_acc, 'fina_valid_acc': valid_acc, 'final_test_acc': test_acc})
    print(f"Final train acc: {train_acc} | Final valid acc: {valid_acc} | Final test acc: {test_acc}")
    # list_test_acc.append(test_acc)
    # list_valid_acc.append(valid_acc)
    # list_train_acc.append(train_acc)

    del trnr
    torch.cuda.empty_cache()
    gc.collect()

    wandb.finish()

    return train_acc, valid_acc, test_acc

    ## record training data
    # print(
    #     "mean and std of test acc: {:.4f} {:.4f} ".format(
    #         np.mean(list_test_acc) * 100, np.std(list_test_acc) * 100
    #     )
    # )

    # try:
    #     to_save = dict(
    #         seed=seed,
    #         test_acc=list_test_acc,
    #         val_acc=list_valid_acc,
    #         train_loss=list_train_loss,
    #         mean_test_acc=np.mean(list_test_acc),
    #         std_test_acc=np.std(list_test_acc),
    #     )
    #     with open(path_json, "w") as f:
    #         json.dump(to_save, f)
    # except:
    #     pass
    # print(
    #     "final mean and std of test acc: ",
    #     f"{np.mean(list_test_acc)*100:.4f} $\\pm$ {np.std(list_test_acc)*100:.4f}",
    # )

def tune_pipe(config):
    train_acc, valid_acc, test_acc = pipe(config)
    tune.report(train_acc=train_acc, valid_acc=valid_acc, test_acc=test_acc)

def run_ray(config:dict):
    exp = 71
    num_samples = 1

    searchSpace = {
        'lr': tune.grid_search([1e-2, 1e-3, 1e-4]),
        'weight_decay': tune.grid_search([0.0, 1e-4, 1e-5]),
        'dropout': tune.grid_search([0.0, 0.2, 0.5, 0.8]),
        'dim_hidden': tune.grid_search([128, 256, 512]),
        'num_layers': tune.grid_search([4, 8, 12]),
        'SLE_threshold': tune.grid_search([0.5, 0.7, 0.9]),
        'random_seed': tune.grid_search([0, 1, 2, 3, 5]),
        'init': 'nimfor',
        'exp': exp
    }
    config.update(searchSpace)
    
    print(config)

    analysis=tune.run(tune_pipe, config=config, name=f"{exp}", num_samples=num_samples, \
        resources_per_trial={'cpu': 12, 'gpu':1}, log_to_file=f"out.log", \
        local_dir="/mnt/jiahanli/nim_output", max_failures=3)

def run_test(config:dict):

    searchSpace = {
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'dropout': 1e-1,
        'dim_hidden': 512,
        'SLE_threshold': 0.5,
        'num_layers': 1,
        'random_seed': 0,
        'init': 'xav',
        'exp': 502
    }
    config.update(searchSpace)

    pipe(config)

if __name__ == '__main__':
    args = BaseOptions().initialize()
    config = vars(args)
    fixConfig = {
        'type_model': 'EnGCN',
        'dataset': 'ogbn-arxiv',
        'epochs': 100,
        'N_exp': 1,
        'batch_size': 10000,
        'tosparse': True,
    }
    config.update(fixConfig)

    run='ray'
    
    if run == 'test':
        run_test(config)
    elif run == 'ray':
        run_ray(config)

    print(1)
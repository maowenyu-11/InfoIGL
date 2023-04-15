import torch
import numpy as np
import os
import random
from texttable import Texttable


def set_seed(seed):

    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False


import argparse


def parse_args():

    str2bool = lambda x: x.lower() == "true"

    parser = argparse.ArgumentParser()
    parser.add_argument('--mol', type=str2bool, default=True)
    parser.add_argument('--global_', type=str2bool, default=True)
    parser.add_argument('--local', type=str2bool, default=True)
    # parser.add_argument('--task_type',
    #                     type=str,
    #                     default="'Multi-label classification")
    parser.add_argument('--pretrain', type=int, default=70)
    parser.add_argument('--hidden_in', type=int, default=-1)
    parser.add_argument('--eval_name', type=str, default="ogbg-molhiv")
    parser.add_argument('--eval_metric', type=str, default="rocauc")
    parser.add_argument('--shift', type=str, default="concept")
    parser.add_argument('--domain', type=str, default="scaffold")
    # parser.add_argument('--step_size', type=float, default=0.001)
    parser.add_argument('--min_lr', type=float, default=1e-8)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--cls_layer', type=int, default=2)
    parser.add_argument('--constraint', type=float, default=0.7)
    parser.add_argument('--pred', type=float, default=0.5)
    parser.add_argument('--g', type=float, default=1.0)
    parser.add_argument('--l', type=float, default=0.5)
    parser.add_argument('--trails', type=int, default=5)
    parser.add_argument('--root', type=str, default="dataset")
    parser.add_argument('--dataset',
                        type=str,
                        default="lbap_core_ic50_scaffold")
    parser.add_argument('--hidden', type=int, default=300)
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--global_pool', type=str, default="sum")
    args = parser.parse_args()
    # print_args(args)
    # setup_seed(args.seed)
    return args


def print_args(args, str_num=80):
    for arg, val in args.__dict__.items():
        print(arg + '.' * (str_num - len(arg) - len(str(val))) + str(val))
    print()


def size_split_idx(dataset, mode):

    num_graphs = len(dataset)
    num_val = int(0.1 * num_graphs)
    num_test = int(0.1 * num_graphs)
    num_train = num_graphs - num_test - num_val

    num_node_list = []
    train_idx = []
    valtest_list = []

    for data in dataset:
        num_node_list.append(data.num_nodes)

    sort_list = np.argsort(num_node_list)

    if mode == 'ls':
        train_idx = sort_list[2 * num_val:]
        valid_test_idx = sort_list[:2 * num_val]
    else:
        train_idx = sort_list[:-2 * num_val]
        valid_test_idx = sort_list[-2 * num_val:]
    random.shuffle(valid_test_idx)
    valid_idx = valid_test_idx[:num_val]
    test_idx = valid_test_idx[num_val:]

    split_idx = {
        'train': torch.tensor(train_idx, dtype=torch.long),
        'valid': torch.tensor(valid_idx, dtype=torch.long),
        'test': torch.tensor(test_idx, dtype=torch.long)
    }
    return split_idx
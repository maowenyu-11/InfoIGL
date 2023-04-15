import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import tensor
from drugood.datasets.builder import build_dataset
from mmcv import Config
from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
from torch_geometric.data import DataLoader
# from torch_geometric.datasets import TUDataset
from torch_geometric.nn import global_mean_pool
from configs.drugood_dataset import DrugOOD
from torch.optim import Adam
from GOOD.data.good_datasets.good_hiv import GOODHIV
from GOOD.data.good_datasets.good_cmnist import GOODCMNIST
from GOOD.data.good_datasets.good_motif import GOODMotif
from util import print_args, parse_args, set_seed, size_split_idx
from model import SSL
import time
import warnings

warnings.filterwarnings('ignore')


def eval(model, loader, device, args, evaluator):
    model.eval()
    # eval_random = args.eval_random
    if args.eval_metric == "acc":
        correct = 0
        for data in loader:
            data = data.to(device)
            if data.x.shape[0] == 1:
                pass
            else:
                with torch.no_grad():
                    pred = model.eval_forward(data)

                    pred = pred.max(1)[1]

                    correct += pred.eq(data.y.view(-1)).sum().item()

        output = correct / len(loader.dataset)

    elif args.eval_metric == "rocauc":
        y_true, preds = [], []

        for data in loader:
            data = data.to(device)
            if data.x.shape[0] == 1:
                pass
            else:
                with torch.no_grad():
                    pred = model.eval_forward(data)

                    pred = torch.max(pred, 1, keepdim=True)[0]

                    y_true.append(data.y.view(pred.shape).detach().cpu())

                    preds.append(pred.detach().cpu())

        y_true = torch.cat(y_true, dim=0).numpy()
        preds = torch.cat(preds, dim=0).numpy()
        input_dict = {"y_true": y_true, "y_pred": preds}
        # try:
        output = evaluator.eval(input_dict)[args.eval_metric]
        # except:
        #     pdb.set_trace()
    else:
        assert False

    return output


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss().to(device)


def main(args, trail):
    set_seed(args.seed)

    edge_dim = -1
    ### automatic dataloading and splitting
    if args.dataset.lower().startswith('lbap'):
        #lbap_core_ic50_assay.json
        config_path = "/data/maowy/SSLGL/configs/" + args.dataset + ".py"
        cfg = Config.fromfile(config_path)
        root = "dataset"

        train_dataset = DrugOOD(root=root,
                                dataset=build_dataset(cfg.data.train),
                                name=args.dataset,
                                mode="train")
        val_dataset = DrugOOD(root=root,
                              dataset=build_dataset(cfg.data.ood_val),
                              name=args.dataset,
                              mode="ood_val")
        test_dataset = DrugOOD(root=root,
                               dataset=build_dataset(cfg.data.ood_test),
                               name=args.dataset,
                               mode="ood_test")

        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True)
        valid_loader = DataLoader(val_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False)
        test_loader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False)
        args.hidden_in = 39
        args.edge_dim = 10
        args.num_classes = 2
        args.mol = True
        args.eval_metric = "rocauc"
        args.eval_name = "ogbg-molhiv"

    elif args.dataset.lower().startswith('ogbg'):
        dataset = PygGraphPropPredDataset(name="ogbg-molbbbp", root=args.root)
        args.num_classes = 2
        args.eval_metric = "rocauc"
        args.eval_name = "ogbg-molbbbp"
        args.hidden_in = 9
        args.mol = True
        if args.domain == "scaffold":
            split_idx = dataset.get_idx_split()
        else:
            split_idx = size_split_idx(dataset, "ls")
        train_loader = DataLoader(dataset[split_idx["train"]],
                                  batch_size=args.batch_size,
                                  shuffle=True)
        valid_loader = DataLoader(dataset[split_idx["valid"]],
                                  batch_size=args.batch_size,
                                  shuffle=False)
        test_loader = DataLoader(dataset[split_idx["test"]],
                                 batch_size=args.batch_size,
                                 shuffle=False)

    elif args.dataset == "hiv":
        dataset, meta_info = GOODHIV.load(args.root,
                                          domain=args.domain,
                                          shift=args.shift,
                                          generate=False)
        train_loader = DataLoader(
            dataset["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )
        valid_loader = DataLoader(
            dataset["val"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )

        test_loader = DataLoader(
            dataset["test"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        args.num_classes = 2
        args.hidden_in = meta_info['dim_node']
        args.eval_metric = "rocauc"
        args.eval_name = "ogbg-molhiv"
        args.mol = True
        # args.task_type = dataset['task']

    elif args.dataset == "cmnist":
        dataset, meta_info = GOODCMNIST.load(args.root,
                                             domain=args.domain,
                                             shift=args.shift,
                                             generate=False)
        train_loader = DataLoader(
            dataset["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )
        valid_loader = DataLoader(
            dataset["val"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )

        test_loader = DataLoader(
            dataset["test"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        args.num_classes = meta_info['num_classes']
        args.eval_metric = "acc"
        args.hidden_in = meta_info['dim_node']
        args.mol = False
        # args.task_type = dataset['task']

    elif args.dataset == "motif":
        dataset, meta_info = GOODMotif.load(args.root,
                                            domain=args.domain,
                                            shift=args.shift,
                                            generate=False)
        train_loader = DataLoader(
            dataset["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )
        valid_loader = DataLoader(
            dataset["val"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )

        test_loader = DataLoader(
            dataset["test"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        args.num_classes = meta_info['num_classes']
        args.hidden_in = meta_info['dim_node']
        args.eval_metric = "acc"
        args.mol = False
        # args.task_type = dataset['task']

    else:
        assert False
    # log
    model = SSL(args, device).to(device)
    evaluator = Evaluator(args.eval_name)

    optimizer = Adam(model.parameters(),
                     lr=args.lr,
                     weight_decay=args.weight_decay)
    lr_scheduler = CosineAnnealingLR(optimizer,
                                     T_max=args.epochs,
                                     eta_min=args.min_lr,
                                     last_epoch=-1,
                                     verbose=False)

    results = {'highest_valid': 0, 'update_test': 0, 'update_epoch': 0}
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        start_time_local = time.time()
        train_loss, loss_pred, loss_global, loss_local = train_causal_epoch(
            model, optimizer, train_loader, device, criterion, args)

        valid_result = eval(model, valid_loader, device, args, evaluator)
        test_result = eval(model, test_loader, device, args, evaluator)
        lr_scheduler.step()
        if valid_result > results['highest_valid'] and epoch > args.pretrain:
            results['highest_valid'] = valid_result
            results['update_test'] = test_result
            results['update_epoch'] = epoch

        print("-" * 150)
        print(
            "Causal | dataset:[{}] fold:[{}] | Epoch:[{}/{}] Loss:[{:.4f}={:.4f}+{:.4f}+{:.4f}] Valid:[{:.4f}] Test:[{:.4f}] | Best Valid:[{:.4f}] Update test:[{:.4f}] at Epoch:[{}] | epoch time[{:.2f}min "
            .format(args.dataset, trail, epoch, args.epochs, train_loss,
                    loss_pred, loss_global, loss_local, valid_result,
                    test_result, results['highest_valid'],
                    results['update_test'], results['update_epoch'],
                    (time.time() - start_time_local) / 60))

        print("-" * 150)
    total_time = time.time() - start_time

    print(
        "mwy: Causal fold:[{}] | Dataset:[{}] | Update Test:[{:.4f}] at epoch [{}] | Total time:{}"
        .format(trail, args.dataset, results['update_test'],
                results['update_epoch'],
                time.strftime('%H:%M:%S', time.gmtime(total_time))))

    print("-" * 150)
    print('\n')
    # final_test_iid.append(results['update_test_iid'])
    return results['update_test']


def config_and_run(args):

    print_args(args)
    # set_seed(args.seed)
    final_test = []
    for trail in range(args.trails):
        test_result = main(args, trail + 1)
        final_test.append(test_result)
    print("mwy finall: Test result: [{:.2f}±{:.2f}]".format(
        np.mean(final_test) * 100,
        np.std(final_test) * 100))
    print("ALL OOD:{}".format(final_test))


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train_causal_epoch(model, optimizer, loader, device, criterion, args):
    model.train()
    total_loss = 0
    total_loss_p = 0
    total_loss_global = 0
    total_loss_local = 0

    for step, data in enumerate(loader):

        optimizer.zero_grad()
        data = data.to(device)

        pre_loss, global_loss, local_loss = model.forward(data, criterion)

        loss = args.pred * pre_loss + args.g * global_loss + args.l * local_loss
        loss.backward()

        total_loss += loss.item() * num_graphs(data)
        total_loss_p += pre_loss.item() * num_graphs(data)
        total_loss_local += local_loss * num_graphs(data)
        total_loss_global += global_loss * num_graphs(data)

        optimizer.step()

    num = len(loader.dataset)
    total_loss = total_loss / num
    total_loss_p = total_loss_p / num
    total_loss_local = total_loss_local / num
    total_loss_global = total_loss_global / num
    # correct_o = correct_o / num
    return total_loss, total_loss_p, total_loss_global, total_loss_local


if __name__ == "__main__":
    args = parse_args()
    config_and_run(args)
    print(
        "settings | constraint:[{}]  pred:[{}] global_co:[{}] local_co:[{}] global/local:[{}/{}]  hidden_in:[{}] num_classes:[{}] batch_size:[{}]  hidden:[{}] lr:[{}] min_lr:[{}] weight_decay[{}] "
        .format(str(args.constraint), str(args.pred), str(args.g), str(args.l),
                str(args.global_), str(args.local), str(args.hidden_in),
                str(args.num_classes), str(args.batch_size), str(args.hidden),
                str(args.lr), str(args.min_lr), str(args.weight_decay)))

    print("-" * 150)

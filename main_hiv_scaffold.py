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
from util import print_args, parse_args, set_seed, size_split_idx
from model_1 import HivCausal
import time
import warnings

warnings.filterwarnings('ignore')


def eval(model, loader, device, args, evaluator):
    model.eval()
    # eval_random = args.eval_random

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

    return output


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss().to(device)
BCELoss = torch.nn.BCEWithLogitsLoss()


def main(args, trail):
    set_seed(args.seed)

    edge_dim = -1
    prototype = None
    memory_bank = torch.randn(args.num_classes, 10, args.hidden)

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

    model = HivCausal(args.hidden, args.num_classes - 1, args.hidden,
                      args.layers).to(device)
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
        model.train()
        total_loss = 0
        total_loss_p = 0
        total_loss_global = 0
        total_loss_local = 0

        for step, data in enumerate(train_loader):

            optimizer.zero_grad()
            data = data.to(device)

            causal = model.forward_causal(data)
            pred = model(causal)
            is_labeled = data.y == data.y
            pred_loss = BCELoss(
                pred.to(torch.float32)[is_labeled],
                data.y.to(torch.float32)[is_labeled])

            if args.global_ == False and args.local == False:
                global_loss = 0
                local_loss = 0
            else:
                class_causal, lack_class = class_split(data.y, causal, args)
                prototype = prototype_update(prototype, args.num_classes,
                                             class_causal, lack_class)
            if args.global_ == True:
                global_loss = global_ssl(prototype, class_causal, lack_class)
            else:
                global_loss = 0
            if args.local == True:
                local_loss, memory_bank = local_ssl(prototype, memory_bank,
                                                    args, class_causal,
                                                    lack_class)
            else:
                local_loss = 0

            loss = args.pred * pred_loss + args.g * global_loss + args.l * local_loss
            loss.backward()

            total_loss += loss.item() * num_graphs(data)
            total_loss_p += pred_loss.item() * num_graphs(data)
            total_loss_local += local_loss * num_graphs(data)
            total_loss_global += global_loss * num_graphs(data)

            optimizer.step()

        num = len(train_loader.dataset)
        total_loss = total_loss / num
        total_loss_p = total_loss_p / num
        total_loss_local = total_loss_local / num
        total_loss_global = total_loss_global / num

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
            .format(args.dataset, trail, epoch, args.epochs, total_loss,
                    total_loss_p, total_loss_global, total_loss_local,
                    valid_result, test_result, results['highest_valid'],
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


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def class_split(y, causal_feature, args):
    class_causal = {}
    lack_class = []

    idx = 0
    for i in range(args.num_classes):
        k = np.where(y.view(-1).cpu() == i)
        if len(k[0]) == 0:
            class_causal[i] = torch.randn(1, int(args.hidden)).cuda()
            lack_class.append(i)
        else:
            class_idx = torch.tensor(k).view(-1)
            class_causal_feature = causal_feature[class_idx]
            class_causal[i] = class_causal_feature

    return class_causal, lack_class


def softmax_with_temperature(input, t=1, axis=-1):
    ex = torch.exp(input / t)
    sum = torch.sum(ex, axis=axis)
    return ex / sum


#update prototypes
def prototype_update(prototype, num_classes, class_causal, lack_class):

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    if prototype == None:
        prototype_ = [
            torch.mean(class_causal[key].to(torch.float32),
                       dim=0,
                       keepdim=True) for key in class_causal
        ]
        prototype = torch.cat(prototype_, dim=0).detach()

    else:
        for i in range(num_classes):

            if i in lack_class:
                continue
            else:

                cosine = cos(prototype[i].detach(), class_causal[i]).detach()
                weights_proto = softmax_with_temperature(cosine, t=5).reshape(
                    1, -1).detach()
                prototype[i] = torch.mm(weights_proto,
                                        class_causal[i]).detach()
    return prototype
    # prototype = torch.cat(prototype, dim=0)


def global_ssl(prototype, class_causal, lack_class):
    distance = None
    for i in range(2):
        if i in lack_class:
            continue
        else:

            prototype_ = torch.cat(
                (prototype[i:i + 1].detach(), prototype[0:i].detach(),
                 prototype[i + 1:].detach()), 0)
            distance_ = torch.einsum('nc,kc->nk', [
                nn.functional.normalize(class_causal[i], dim=1),
                nn.functional.normalize(prototype_, dim=1)
            ])
            #distance_ /= 5
            if distance is None:
                distance = F.softmax(distance_, dim=1)
            else:
                distance = torch.cat((distance, F.softmax(distance_, dim=1)),
                                     0)
    labels = torch.zeros(distance.shape[0], dtype=torch.long).cuda()

    loss = criterion(distance, labels)
    return loss


def local_ssl(prototype, memory_bank, args, class_causal, lack_class):
    nce = None
    for key in class_causal:
        class_causal[key] = args.constraint * class_causal[key].float() + (
            1 - args.constraint) * prototype[key].float().detach()

    for i in lack_class:
        _ = class_causal.pop(i)

    for i in range(2):
        if i in lack_class:
            continue
        else:
            pos = class_causal[i][0:1]
            class_causal_ = class_causal.copy()
            _ = class_causal_.pop(i)
            if class_causal_ == {}:
                hard_neg = memory_bank[i].detach()
            else:
                neg = torch.cat(list(class_causal_.values()))
                # prototype_=self.prototype.clone().detach()
                distance = F.softmax(torch.einsum(
                    'kc,nc->kn', [prototype[i:i + 1].detach(), neg]),
                                     dim=1)
                dis, idx = torch.sort(distance)
                if len(idx[0]) < 10:
                    hard_neg = torch.cat(
                        (memory_bank[i][0:(10 - len(distance[0])), :].detach(),
                         neg), 0)

                else:
                    hard_neg = neg[idx[0][0:10]]

                memory_bank[i] = hard_neg
            sample = torch.cat((pos, hard_neg), 0)
            nce_ = F.softmax(torch.einsum('nc,kc->nk',
                                          [class_causal[i], sample]),
                             dim=1)
            if nce is None:
                nce = nce_
            else:
                nce = torch.cat((nce, nce_), 0)

    labels = torch.zeros(nce.shape[0], dtype=torch.long).cuda()
    loss = criterion(nce, labels)
    return loss, memory_bank


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

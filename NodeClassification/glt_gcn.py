import os
import time

from scipy import spatial
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import argparse
import numpy
import torch
import torch.nn as nn
import numpy as np
import net as net
from utils import load_data
from sklearn.metrics import f1_score
import pruning.pruning_gcn as pruning
import pruning.adv_pruning_gcn as adv_pruning
import copy
import warnings
warnings.filterwarnings('ignore')
numpy.set_printoptions(threshold=numpy.inf)

# retrain
def run_fix_mask(args, seed, rewind_weight_mask, p):
    pruning.setup_seed(seed)
    adj, features, labels, idx_train, idx_val, idx_test = load_data(args['dataset'])
    adj = adj.cuda()
    features = features.cuda()
    labels = labels.cuda()
    loss_func = nn.CrossEntropyLoss()
    net_gcn = net.net_gcn(embedding_dim=args['embedding_dim'], adj=adj.cuda())
    pruning.add_mask(net_gcn)
    net_gcn = net_gcn.cuda()
    net_gcn.load_state_dict(rewind_weight_mask)
    adj_spar, wei_spar = pruning.print_sparsity(net_gcn)
    for name, param in net_gcn.named_parameters():
        if 'mask' in name:
            param.requires_grad = False
    optimizer = torch.optim.Adam(net_gcn.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    best_val_acc = {'val_acc': 0, 'epoch': 0, 'test_acc': 0}
    acc_list = []
    loss_list = []
    for epoch in range(args['fix_epochs']):
        optimizer.zero_grad()
        output = net_gcn(features, adj)
        loss = loss_func(output[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            output = net_gcn(features, adj, val_test=True)
            acc_val = f1_score(labels[idx_val].cpu().numpy(), output[idx_val].cpu().numpy().argmax(axis=1),
                               average='micro')
            acc_test = f1_score(labels[idx_test].cpu().numpy(), output[idx_test].cpu().numpy().argmax(axis=1),
                                average='micro')
            acc_list.append(acc_test)
            loss_list.append(acc_val)
            if acc_val > best_val_acc['val_acc']:
                best_val_acc['val_acc'] = acc_val
                best_val_acc['test_acc'] = acc_test
                best_val_acc['epoch'] = epoch
    return best_val_acc['val_acc'], best_val_acc['test_acc'], best_val_acc['epoch'], adj_spar, wei_spar


def reverse_adj_mask(rewind_weight_mask, adj):
    ones = torch.ones_like(rewind_weight_mask)
    zeros = torch.zeros_like(rewind_weight_mask)
    mask = torch.where(rewind_weight_mask == 0.0, ones, zeros)
    mask = torch.where(adj.cpu() == 0.0, zeros, mask)
    return mask

def reverse_weight_mask(rewind_weight_mask):
    ones = torch.ones_like(rewind_weight_mask)
    zeros = torch.zeros_like(rewind_weight_mask)
    mask = torch.where(rewind_weight_mask == 0.0, ones, zeros)
    return mask

def reverse_mask(mask_weight_tensor, adj=None):
    ones = torch.ones_like(mask_weight_tensor)
    zeros = torch.zeros_like(mask_weight_tensor)
    if adj:
        mask = torch.where(mask_weight_tensor == 0 & adj > 0.5, ones, zeros)
    else:
        mask = torch.where(mask_weight_tensor == 0, ones, zeros)
    return mask

# get sparse model
def run_get_mask(args, seed, imp_num, rewind_weight_mask=None):
    pruning.setup_seed(seed)
    adj, features, labels, idx_train, idx_val, idx_test = load_data(args['dataset'])
    adj = adj.cuda()
    features = features.cuda()
    labels = labels.cuda()
    loss_func = nn.CrossEntropyLoss()
    net_gcn = net.net_gcn(embedding_dim=args['embedding_dim'], adj=adj)
    pruning.add_mask(net_gcn)
    net_gcn = net_gcn.cuda()
    if args['weight_dir']:
        print("load : {}".format(args['weight_dir']))
        encoder_weight = {}
        cl_ckpt = torch.load(args['weight_dir'], map_location='cuda')
        encoder_weight['weight_orig_weight'] = cl_ckpt['gcn.fc.weight']
        ori_state_dict = net_gcn.net_layer[0].state_dict()
        ori_state_dict.update(encoder_weight)
        net_gcn.net_layer[0].load_state_dict(ori_state_dict)
    if rewind_weight_mask:
        net_gcn.load_state_dict(rewind_weight_mask)
        if not args['rewind_soft_mask'] or args['init_soft_mask_type'] == 'all_one':
            pruning.soft_mask_init(net_gcn, args['init_soft_mask_type'], seed)
    else:
        pruning.soft_mask_init(net_gcn, args['init_soft_mask_type'], seed)

    optimizer = torch.optim.Adam(net_gcn.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    best_val_acc = {'val_acc': 0, 'epoch': 0, 'test_acc': 0}
    rewind_weight = copy.deepcopy(net_gcn.state_dict())
    for epoch in range(args['mask_epochs']):
        optimizer.zero_grad()
        output = net_gcn(features, adj)
        loss = loss_func(output[idx_train], labels[idx_train])
        loss.backward()
        pruning.subgradient_update_mask(net_gcn, args)  # l1 norm
        optimizer.step()
        with torch.no_grad():
            output = net_gcn(features, adj, val_test=True)
            acc_val = f1_score(labels[idx_val].cpu().numpy(), output[idx_val].cpu().numpy().argmax(axis=1),
                               average='micro')
            acc_test = f1_score(labels[idx_test].cpu().numpy(), output[idx_test].cpu().numpy().argmax(axis=1),
                                average='micro')
            if acc_val > best_val_acc['val_acc']:
                best_val_acc['test_acc'] = acc_test
                best_val_acc['val_acc'] = acc_val
                best_val_acc['epoch'] = epoch
                weight_dict, best_epoch_mask = pruning.get_final_mask_epoch(net_gcn, adj_percent=args['pruning_percent_adj'], wei_percent=args['pruning_percent_wei'])
    return best_epoch_mask, rewind_weight

def parser_loader():
    parser = argparse.ArgumentParser(description='GLT')
    ###### Unify pruning settings #######
    parser.add_argument('--s1', type=float, default=0.0001, help='scale sparse rate (default: 0.0001)')
    parser.add_argument('--s2', type=float, default=0.0001, help='scale sparse rate (default: 0.0001)')
    parser.add_argument('--mask_epochs', type=int, default=200)
    parser.add_argument('--fix_epochs', type=int, default=200)
    parser.add_argument('--pruning_percent_wei', type=float, default=0.1)
    parser.add_argument('--pruning_percent_adj', type=float, default=0.1)
    parser.add_argument('--weight_dir', type=str, default='')
    parser.add_argument('--rewind_soft_mask', action='store_true')
    parser.add_argument('--init_soft_mask_type', type=str, default='', help='all_one, kaiming, normal, uniform')
    ###### Others settings #######
    parser.add_argument('--dataset', type=str, default='citeseer')
    parser.add_argument('--filename', type=str, default='Result/test.txt')
    parser.add_argument('--embedding-dim', nargs='+', type=int, default=[3703, 16, 6])
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=200)
    return parser

# Refinement
def adv_train(args, seed, rewind_weight):
    pruning.setup_seed(seed)
    adj, features, labels, idx_train, idx_val, idx_test = load_data(args['dataset'])
    adj = adj.cuda()
    features = features.cuda()
    labels = labels.cuda()
    loss_func = nn.CrossEntropyLoss()
    rewind_weight_mask_copy = copy.deepcopy(rewind_weight)
    rewind_weight_mask_copy['net_layer.0.weight_mask_train'] = reverse_weight_mask(rewind_weight['net_layer.0.weight_mask_train'])
    rewind_weight_mask_copy['net_layer.0.weight_mask_fixed'] = reverse_weight_mask(rewind_weight['net_layer.0.weight_mask_fixed'])
    rewind_weight_mask_copy['net_layer.1.weight_mask_train'] = reverse_weight_mask(rewind_weight['net_layer.1.weight_mask_train'])
    rewind_weight_mask_copy['net_layer.1.weight_mask_fixed'] = reverse_weight_mask(rewind_weight['net_layer.1.weight_mask_fixed'])
    best_val_history = {'val_acc': 0, 'round': 0, 'epoch': 0, 'test_acc': 0}
    best_dict_history = {}
    # hyparameters
    rounds = 6
    origin_weight_temp = copy.deepcopy(rewind_weight)
    adv_weight_temp = copy.deepcopy(rewind_weight_mask_copy)
    model_origin = net.net_gcn(embedding_dim=args['embedding_dim'], adj=adj.cuda())  # 提取的彩票模型
    model_adv = net.net_gcn(embedding_dim=args['embedding_dim'], adj=adj.cuda())  # 准备进行对抗训练的反彩票模型
    pruning.add_mask(model_origin)
    pruning.add_mask(model_adv)
    model_origin = model_origin.cuda()
    model_adv = model_adv.cuda()
    for round in range(rounds):
        if round != 0:
            origin_weight_temp['net_layer.0.weight_mask_train'] = best_mask_refine_origin['weight1_mask']
            origin_weight_temp['net_layer.0.weight_mask_fixed'] = best_mask_refine_origin['weight1_mask']
            origin_weight_temp['net_layer.1.weight_mask_train'] = best_mask_refine_origin['weight2_mask']
            origin_weight_temp['net_layer.1.weight_mask_fixed'] = best_mask_refine_origin['weight2_mask']
            adv_weight_temp['net_layer.0.weight_mask_train'] = best_mask_refine_adv['weight1_mask']
            adv_weight_temp['net_layer.0.weight_mask_fixed'] = best_mask_refine_adv['weight1_mask']
            adv_weight_temp['net_layer.1.weight_mask_train'] = best_mask_refine_adv['weight2_mask']
            adv_weight_temp['net_layer.1.weight_mask_fixed'] = best_mask_refine_adv['weight2_mask']
        model_origin.load_state_dict(origin_weight_temp)
        model_adv.load_state_dict(adv_weight_temp)
        adj_spar, wei_spar = pruning.get_sparsity(model_origin)
        recover_rate = 0.005 * (100 - wei_spar) / 100
        pruning.soft_mask_init(model_origin, args['init_soft_mask_type'], seed)
        pruning.soft_mask_init(model_adv, args['init_soft_mask_type'], seed)
        optimizer_origin = torch.optim.Adam(model_origin.parameters(), lr=args['lr'],weight_decay=args['weight_decay'])
        optimizer_adv = torch.optim.Adam(model_adv.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
        best_val_acc_origin = {'val_acc': 0, 'epoch': 0, 'test_acc': 0}
        best_val_acc_adv = {'val_acc': 0, 'epoch': 0, 'test_acc': 0}
        for epoch in range(20):
            optimizer_origin.zero_grad()
            optimizer_adv.zero_grad()
            output_origin = model_origin(features, adj)
            output_adv = model_adv(features, adj)
            loss_origin = loss_func(output_origin[idx_train], labels[idx_train])
            loss_origin.backward()
            loss_adv = loss_func(output_adv[idx_train], labels[idx_train])
            loss_adv.backward()
            adv_pruning.subgradient_update_mask(model_origin, args)
            adv_pruning.subgradient_update_mask(model_adv, args)
            optimizer_origin.step()
            optimizer_adv.step()
            with torch.no_grad():
                output_origin = model_origin(features, adj, val_test=True)
                output_adv = model_adv(features, adj, val_test=True)
                acc_val_origin = f1_score(labels[idx_val].cpu().numpy(),
                                          output_origin[idx_val].cpu().numpy().argmax(axis=1), average='micro')
                acc_test_origin = f1_score(labels[idx_test].cpu().numpy(),
                                           output_origin[idx_test].cpu().numpy().argmax(axis=1), average='micro')
                acc_val_adv = f1_score(labels[idx_val].cpu().numpy(), output_adv[idx_val].cpu().numpy().argmax(axis=1),
                                       average='micro')
                acc_test_adv = f1_score(labels[idx_test].cpu().numpy(),
                                        output_adv[idx_test].cpu().numpy().argmax(axis=1),
                                        average='micro')
                if acc_val_adv > best_val_acc_adv['val_acc']:
                    best_val_acc_adv['test_acc'] = acc_test_adv
                    best_val_acc_adv['val_acc'] = acc_val_adv
                    best_val_acc_adv['epoch'] = epoch
                    wei_thre_index, best_epoch_mask_adv = adv_pruning.get_final_mask_epoch(model_adv, recover_rate)
                    best_adv_model_copy = copy.deepcopy(model_adv)
                if acc_val_origin > best_val_acc_origin['val_acc']:
                    best_val_acc_origin['test_acc'] = acc_test_origin
                    best_val_acc_origin['val_acc'] = acc_val_origin
                    best_val_acc_origin['epoch'] = epoch
                    best_epoch_mask_origin = adv_pruning.get_final_mask_epoch_adv(model_origin, wei_thre_index)
                    best_origin_model_copy = copy.deepcopy(model_origin)
        best_mask_refine_origin = adv_pruning.get_final_mask_round_origin(best_origin_model_copy,
                                                                          best_epoch_mask_adv,
                                                                          best_epoch_mask_origin)
        best_mask_refine_adv = adv_pruning.get_final_mask_round_adv(best_adv_model_copy,
                                                                    best_epoch_mask_adv,
                                                                    best_epoch_mask_origin)
        if round != 0:
            now_origin_weight_flatten_l0 = np.array(best_epoch_mask_origin['weight1_mask'].flatten().cpu())
            similarity_origin_adv = 1 - spatial.distance.cosine(last_adv_weight_flatten_l0, now_origin_weight_flatten_l0)
            if similarity_origin_adv > 0.6:
                recover_rate = recover_rate * (1-recover_rate) * 2
                wei_thre_index, best_epoch_mask_adv = adv_pruning.get_final_mask_epoch(last_best_adv_model_copy,recover_rate)
                best_epoch_mask_origin = adv_pruning.get_final_mask_epoch_adv(last_best_origin_model_copy, wei_thre_index)
                best_mask_refine_origin = adv_pruning.get_final_mask_round_origin(last_best_adv_model_copy,
                                                                                  best_epoch_mask_adv,
                                                                                  best_epoch_mask_origin)
                best_mask_refine_adv = adv_pruning.get_final_mask_round_adv(last_best_adv_model_copy,
                                                                            best_epoch_mask_adv,
                                                                            best_epoch_mask_origin)
                rounds = rounds+1
                continue
            else:
                pass
        last_adv_weight_flatten_l0 = np.array(best_epoch_mask_adv['weight1_mask'].flatten().cpu())
        last_best_adv_model_copy = copy.deepcopy(best_adv_model_copy)
        last_best_origin_model_copy = copy.deepcopy(best_origin_model_copy)
        if best_val_acc_origin['val_acc'] > best_val_history['val_acc']:
            best_val_history['test_acc'] = best_val_acc_origin['test_acc']
            best_val_history['val_acc'] = best_val_acc_origin['val_acc']
            best_val_history['epoch'] = best_val_acc_origin['epoch']
            best_val_history['round'] = round
            best_dict_history = copy.deepcopy(best_mask_refine_origin)
    rewind_weight['net_layer.0.weight_mask_train'] = best_dict_history['weight1_mask']
    rewind_weight['net_layer.0.weight_mask_fixed'] = best_dict_history['weight1_mask']
    rewind_weight['net_layer.1.weight_mask_train'] = best_dict_history['weight2_mask']
    rewind_weight['net_layer.1.weight_mask_fixed'] = best_dict_history['weight2_mask']
    model_origin.load_state_dict(rewind_weight)
    adj_spar, wei_spar = pruning.print_sparsity(model_origin)
    print("Refine Mask : Best Val:[{:.2f}] at round:[{}] epoch:[{}] | Final Test Acc:[{:.2f}] Adj:[{:.2f}%] Wei:[{:.2f}%]".format(best_val_history['val_acc'] * 100, best_val_history['round'], best_val_history['epoch'],
            best_val_history['test_acc'] * 100, adj_spar, wei_spar))
    best_acc_val, final_acc_test, final_epoch_list, adj_spar, wei_spar = run_fix_mask(args, seed, rewind_weight, p)
    return best_acc_val, final_acc_test, final_epoch_list, adj_spar, wei_spar


if __name__ == "__main__":
    parser = parser_loader()
    args = vars(parser.parse_args())
    print(args)
    filename = args['filename']
    seed_dict = {'cora': 2377, 'citeseer': 4428, 'pubmed': 3333}
    seed = seed_dict[args['dataset']]
    rewind_weight = None
    acc_origin = []
    acc_reverse = []
    model_sparsity = []
    model_sparsity_adv = []
    start = time.clock()
    previous_round = time.clock()
    with open(filename, "w") as f:
        for p in range(20):
            final_mask_dict, rewind_weight = run_get_mask(args, seed, p, rewind_weight)
            rewind_weight['net_layer.0.weight_mask_train'] = final_mask_dict['weight1_mask']
            rewind_weight['net_layer.0.weight_mask_fixed'] = final_mask_dict['weight1_mask']
            rewind_weight['net_layer.1.weight_mask_train'] = final_mask_dict['weight2_mask']
            rewind_weight['net_layer.1.weight_mask_fixed'] = final_mask_dict['weight2_mask']
            print("=" * 120)
            rewind_weight_pass = copy.deepcopy(rewind_weight)
            best_acc_val, final_acc_test, final_epoch_list, adj_spar, wei_spar = adv_train(args, seed, rewind_weight_pass)
            round_time = time.clock()
            print("round--{}: running time--{}s".format(p, round_time-previous_round))
            previous_round = copy.deepcopy(round_time)
            print("Adv-GLT : Sparsity:[{}], Best Val:[{:.2f}] at epoch:[{}] | Final Test Acc:[{:.2f}] Adj:[{:.2f}%] Wei:[{:.2f}%]"
                  .format(p + 1, best_acc_val * 100, final_epoch_list, final_acc_test * 100, adj_spar, wei_spar))
            model_sparsity_adv.append(100 - wei_spar)
            acc_reverse.append(final_acc_test * 100)
            f.write("{:2f},{:2f},{:2f}\n".format((100 - adj_spar), (100 - wei_spar), final_acc_test * 100))
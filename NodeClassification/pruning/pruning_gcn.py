import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch
import torch.nn as nn
from abc import ABC
import numpy as np
import random
import torch.nn.init as init
import math

def prune_adj(oriadj, non_zero_idx, percent):
    original_prune_num = int((non_zero_idx / 2) * (percent / 100))
    adj = np.copy(oriadj)
    low_adj = np.tril(adj, -1)
    non_zero_low_adj = low_adj[low_adj != 0]
    low_pcen = np.percentile(abs(non_zero_low_adj), percent)
    under_threshold = abs(low_adj) < low_pcen
    before = len(non_zero_low_adj)
    low_adj[under_threshold] = 0
    non_zero_low_adj = low_adj[low_adj != 0]
    after = len(non_zero_low_adj)
    rest_pruned = original_prune_num - (before - after)
    if rest_pruned > 0:
        mask_low_adj = (low_adj != 0)
        low_adj[low_adj == 0] = 2000000
        flat_indices = np.argpartition(low_adj.ravel(), rest_pruned - 1)[:rest_pruned]
        row_indices, col_indices = np.unravel_index(flat_indices, low_adj.shape)
        low_adj = np.multiply(low_adj, mask_low_adj)
        low_adj[row_indices, col_indices] = 0
    adj = low_adj + np.transpose(low_adj)
    adj = np.add(adj, np.identity(adj.shape[0]))
    return adj

def setup_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

class AddTrainableMask(ABC):
    _tensor_name: str

    def __init__(self):
        pass

    def __call__(self, module, inputs):
        setattr(module, self._tensor_name, self.apply_mask(module))

    def apply_mask(self, module):
        mask_train = getattr(module, self._tensor_name + "_mask_train")
        mask_fixed = getattr(module, self._tensor_name + "_mask_fixed")
        orig_weight = getattr(module, self._tensor_name + "_orig_weight")
        pruned_weight = mask_train * mask_fixed * orig_weight

        return pruned_weight

    @classmethod
    def apply(cls, module, name, mask_train, mask_fixed, *args, **kwargs):
        method = cls(*args, **kwargs)
        method._tensor_name = name
        orig = getattr(module, name)

        module.register_parameter(name + "_mask_train", mask_train.to(dtype=orig.dtype))
        module.register_parameter(name + "_mask_fixed", mask_fixed.to(dtype=orig.dtype))
        module.register_parameter(name + "_orig_weight", orig)
        del module._parameters[name]

        setattr(module, name, method.apply_mask(module))
        module.register_forward_pre_hook(method)

        return method


def add_mask(model, init_mask_dict=None):
    if init_mask_dict is None:
        mask1_train = nn.Parameter(torch.ones_like(model.net_layer[0].weight))
        mask1_fixed = nn.Parameter(torch.ones_like(model.net_layer[0].weight), requires_grad=False)
        mask2_train = nn.Parameter(torch.ones_like(model.net_layer[1].weight))
        mask2_fixed = nn.Parameter(torch.ones_like(model.net_layer[1].weight), requires_grad=False)
    else:
        mask1_train = nn.Parameter(init_mask_dict['mask1_train'])
        mask1_fixed = nn.Parameter(init_mask_dict['mask1_fixed'], requires_grad=False)
        mask2_train = nn.Parameter(init_mask_dict['mask2_train'])
        mask2_fixed = nn.Parameter(init_mask_dict['mask2_fixed'], requires_grad=False)
    AddTrainableMask.apply(model.net_layer[0], 'weight', mask1_train, mask1_fixed)
    AddTrainableMask.apply(model.net_layer[1], 'weight', mask2_train, mask2_fixed)

def generate_mask(model):
    mask_dict = {}
    mask_dict['mask1'] = torch.zeros_like(model.net_layer[0].weight)
    mask_dict['mask2'] = torch.zeros_like(model.net_layer[1].weight)
    return mask_dict

def subgradient_update_mask(model, args):
    model.adj_mask1_train.grad.data.add_(args['s1'] * torch.sign(model.adj_mask1_train.data))
    model.net_layer[0].weight_mask_train.grad.data.add_(
        args['s2'] * torch.sign(model.net_layer[0].weight_mask_train.data))
    model.net_layer[1].weight_mask_train.grad.data.add_(
        args['s2'] * torch.sign(model.net_layer[1].weight_mask_train.data))

def get_mask_distribution(model, if_numpy=True):
    adj_mask_tensor = model.adj_mask1_train.flatten()
    nonzero = torch.abs(adj_mask_tensor) > 0
    adj_mask_tensor = adj_mask_tensor[nonzero]

    weight_mask_tensor0 = model.net_layer[0].weight_mask_train.flatten()
    nonzero = torch.abs(weight_mask_tensor0) > 0
    weight_mask_tensor0 = weight_mask_tensor0[nonzero]

    weight_mask_tensor1 = model.net_layer[1].weight_mask_train.flatten()
    nonzero = torch.abs(weight_mask_tensor1) > 0
    weight_mask_tensor1 = weight_mask_tensor1[nonzero]

    weight_mask_tensor = torch.cat([weight_mask_tensor0, weight_mask_tensor1])
    if if_numpy:
        return adj_mask_tensor.detach().cpu().numpy(), weight_mask_tensor.detach().cpu().numpy()
    else:
        return adj_mask_tensor.detach().cpu(), weight_mask_tensor.detach().cpu()

def get_each_mask(mask_weight_tensor, threshold):
    ones = torch.ones_like(mask_weight_tensor)
    zeros = torch.zeros_like(mask_weight_tensor)
    mask = torch.where(mask_weight_tensor.abs() > threshold, ones, zeros)
    return mask

def get_each_mask_admm(mask_weight_tensor, threshold):
    zeros = torch.zeros_like(mask_weight_tensor)
    mask = torch.where(mask_weight_tensor.abs() > threshold, mask_weight_tensor, zeros)
    return mask

##### pruning remain mask percent #######
def get_final_mask_epoch(model, adj_percent, wei_percent):
    adj_mask, wei_mask = get_mask_distribution(model, if_numpy=False)
    adj_total = adj_mask.shape[0]
    wei_total = wei_mask.shape[0]
    ### sort
    adj_y, adj_i = torch.sort(adj_mask.abs())
    wei_y, wei_i = torch.sort(wei_mask.abs())
    ### get threshold
    adj_thre_index = int(adj_total * adj_percent)
    adj_thre = adj_y[adj_thre_index]
    wei_thre_index = int(wei_total * wei_percent)
    wei_thre = wei_y[wei_thre_index]
    mask_dict = {}
    weight_dict = {}
    ori_adj_mask = model.adj_mask1_train.detach().cpu()
    weight_dict['adj_weight'] = ori_adj_mask
    weight_dict['model_weight1'] = model.net_layer[0].state_dict()['weight_mask_train']
    weight_dict['model_weight2'] = model.net_layer[1].state_dict()['weight_mask_train']
    mask_dict['adj_mask'] = get_each_mask(ori_adj_mask, adj_thre)
    mask_dict['weight1_mask'] = get_each_mask(model.net_layer[0].state_dict()['weight_mask_train'], wei_thre)
    mask_dict['weight2_mask'] = get_each_mask(model.net_layer[1].state_dict()['weight_mask_train'], wei_thre)
    return weight_dict, mask_dict

def print_sparsity(model):
    adj_nonzero = model.adj_nonzero
    adj_mask_nonzero = model.adj_mask2_fixed.sum().item()
    adj_spar = adj_mask_nonzero * 100 / adj_nonzero
    weight1_total = model.net_layer[0].weight_mask_fixed.numel()
    weight2_total = model.net_layer[1].weight_mask_fixed.numel()
    weight_total = weight1_total + weight2_total
    weight1_nonzero = model.net_layer[0].weight_mask_fixed.sum().item()
    weight2_nonzero = model.net_layer[1].weight_mask_fixed.sum().item()
    weight_nonzero = weight1_nonzero + weight2_nonzero
    wei_spar = weight_nonzero * 100 / weight_total
    print("-" * 100)
    print("Sparsity: Adj:[{:.2f}%] Wei:[{:.2f}%]"
          .format(adj_spar, wei_spar))
    print("-" * 100)
    return adj_spar, wei_spar

def get_sparsity(model):
    adj_nonzero = model.adj_nonzero
    adj_mask_nonzero = model.adj_mask2_fixed.sum().item()
    adj_spar = adj_mask_nonzero * 100 / adj_nonzero
    weight1_total = model.net_layer[0].weight_mask_fixed.numel()
    weight2_total = model.net_layer[1].weight_mask_fixed.numel()
    weight_total = weight1_total + weight2_total
    weight1_nonzero = model.net_layer[0].weight_mask_fixed.sum().item()
    weight2_nonzero = model.net_layer[1].weight_mask_fixed.sum().item()
    weight_nonzero = weight1_nonzero + weight2_nonzero
    wei_spar = weight_nonzero * 100 / weight_total
    return adj_spar, wei_spar

def load_only_mask(model, all_ckpt):
    model_state_dict = model.state_dict()
    masks_state_dict = {k: v for k, v in all_ckpt.items() if 'mask' in k}
    model_state_dict.update(masks_state_dict)
    model.load_state_dict(model_state_dict)

def add_trainable_mask_noise(model, c):
    model.adj_mask1_train.requires_grad = False
    model.net_layer[0].weight_mask_train.requires_grad = False
    model.net_layer[1].weight_mask_train.requires_grad = False
    rand1 = (2 * torch.rand(model.adj_mask1_train.shape) - 1) * c
    rand1 = rand1.to(model.adj_mask1_train.device)
    rand1 = rand1 * model.adj_mask1_train
    model.adj_mask1_train.add_(rand1)
    rand2 = (2 * torch.rand(model.net_layer[0].weight_mask_train.shape) - 1) * c
    rand2 = rand2.to(model.net_layer[0].weight_mask_train.device)
    rand2 = rand2 * model.net_layer[0].weight_mask_train
    model.net_layer[0].weight_mask_train.add_(rand2)
    rand3 = (2 * torch.rand(model.net_layer[1].weight_mask_train.shape) - 1) * c
    rand3 = rand3.to(model.net_layer[1].weight_mask_train.device)
    rand3 = rand3 * model.net_layer[1].weight_mask_train
    model.net_layer[1].weight_mask_train.add_(rand3)
    model.adj_mask1_train.requires_grad = True
    model.net_layer[0].weight_mask_train.requires_grad = True
    model.net_layer[1].weight_mask_train.requires_grad = True

def soft_mask_init(model, init_type, seed):
    setup_seed(seed)
    if init_type == 'all_one':
        add_trainable_mask_noise(model, c=1e-5)
    elif init_type == 'kaiming':
        init.kaiming_uniform_(model.adj_mask1_train, a=math.sqrt(5))
        model.adj_mask1_train.requires_grad = False
        model.adj_mask1_train.mul_(model.adj_mask2_fixed)
        model.adj_mask1_train.requires_grad = True
        init.kaiming_uniform_(model.net_layer[0].weight_mask_train, a=math.sqrt(5))
        model.net_layer[0].weight_mask_train.requires_grad = False
        model.net_layer[0].weight_mask_train.mul_(model.net_layer[0].weight_mask_fixed)
        model.net_layer[0].weight_mask_train.requires_grad = True
        init.kaiming_uniform_(model.net_layer[1].weight_mask_train, a=math.sqrt(5))
        model.net_layer[1].weight_mask_train.requires_grad = False
        model.net_layer[1].weight_mask_train.mul_(model.net_layer[1].weight_mask_fixed)
        model.net_layer[1].weight_mask_train.requires_grad = True
    elif init_type == 'normal':
        mean = 1.0
        std = 0.1
        init.normal_(model.adj_mask1_train, mean=mean, std=std)
        model.adj_mask1_train.requires_grad = False
        model.adj_mask1_train.mul_(model.adj_mask2_fixed)
        model.adj_mask1_train.requires_grad = True
        init.normal_(model.net_layer[0].weight_mask_train, mean=mean, std=std)
        model.net_layer[0].weight_mask_train.requires_grad = False
        model.net_layer[0].weight_mask_train.mul_(model.net_layer[0].weight_mask_fixed)
        model.net_layer[0].weight_mask_train.requires_grad = True
        init.normal_(model.net_layer[1].weight_mask_train, mean=mean, std=std)
        model.net_layer[1].weight_mask_train.requires_grad = False
        model.net_layer[1].weight_mask_train.mul_(model.net_layer[1].weight_mask_fixed)
        model.net_layer[1].weight_mask_train.requires_grad = True
    elif init_type == 'uniform':
        a = 0.8
        b = 1.2
        init.uniform_(model.adj_mask1_train, a=a, b=b)
        model.adj_mask1_train.requires_grad = False
        model.adj_mask1_train.mul_(model.adj_mask2_fixed)
        model.adj_mask1_train.requires_grad = True
        init.uniform_(model.net_layer[0].weight_mask_train, a=a, b=b)
        model.net_layer[0].weight_mask_train.requires_grad = False
        model.net_layer[0].weight_mask_train.mul_(model.net_layer[0].weight_mask_fixed)
        model.net_layer[0].weight_mask_train.requires_grad = True
        init.uniform_(model.net_layer[1].weight_mask_train, a=a, b=b)
        model.net_layer[1].weight_mask_train.requires_grad = False
        model.net_layer[1].weight_mask_train.mul_(model.net_layer[1].weight_mask_fixed)
        model.net_layer[1].weight_mask_train.requires_grad = True
    else:
        assert False





import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import torch.nn as nn
from abc import ABC
import numpy as np
import random

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


def add_mask(model):
    mask1_train = nn.Parameter(torch.ones_like(model.ginlayers[0].apply_func.mlp.linear.weight))
    mask1_fixed = nn.Parameter(torch.ones_like(model.ginlayers[0].apply_func.mlp.linear.weight), requires_grad=False)
    mask2_train = nn.Parameter(torch.ones_like(model.ginlayers[1].apply_func.mlp.linear.weight))
    mask2_fixed = nn.Parameter(torch.ones_like(model.ginlayers[1].apply_func.mlp.linear.weight), requires_grad=False)

    AddTrainableMask.apply(model.ginlayers[0].apply_func.mlp.linear, 'weight', mask1_train, mask1_fixed)
    AddTrainableMask.apply(model.ginlayers[1].apply_func.mlp.linear, 'weight', mask2_train, mask2_fixed)


def generate_mask(model):
    mask_dict = {}
    mask_dict['mask1'] = torch.zeros_like(model.ginlayers[0].apply_func.mlp.linear.weight)
    mask_dict['mask2'] = torch.zeros_like(model.ginlayers[1].apply_func.mlp.linear.weight)

    return mask_dict


def subgradient_update_mask(model, args):
    model.adj_mask1_train.grad.data.add_(args['s1'] * torch.sign(model.adj_mask1_train.data))
    model.ginlayers[0].apply_func.mlp.linear.weight_mask_train.grad.data.add_(
        args['s2'] * torch.sign(model.ginlayers[0].apply_func.mlp.linear.weight_mask_train.data))
    model.ginlayers[1].apply_func.mlp.linear.weight_mask_train.grad.data.add_(
        args['s2'] * torch.sign(model.ginlayers[1].apply_func.mlp.linear.weight_mask_train.data))


def get_mask_distribution(model, if_numpy=True):
    adj_mask_tensor = model.adj_mask1_train.flatten()
    nonzero = torch.abs(adj_mask_tensor) > 0
    adj_mask_tensor = adj_mask_tensor[nonzero]  # 13264 - 2708

    weight_mask_tensor0 = model.ginlayers[0].apply_func.mlp.linear.weight_mask_train.flatten()  # 22928
    nonzero = torch.abs(weight_mask_tensor0) > 0
    weight_mask_tensor0 = weight_mask_tensor0[nonzero]

    weight_mask_tensor1 = model.ginlayers[1].apply_func.mlp.linear.weight_mask_train.flatten()  # 22928
    nonzero = torch.abs(weight_mask_tensor1) > 0
    weight_mask_tensor1 = weight_mask_tensor1[nonzero]

    weight_mask_tensor = torch.cat([weight_mask_tensor0, weight_mask_tensor1])  # 112
    # np.savez('mask', adj_mask=adj_mask_tensor.detach().cpu().numpy(), weight_mask=weight_mask_tensor.detach().cpu().numpy())
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
    # adj_mask.add_((2 * torch.rand(adj_mask.shape) - 1) * 1e-5)
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
    ori_adj_mask = model.adj_mask1_train.detach().cpu()
    mask_dict['adj_mask'] = get_each_mask(ori_adj_mask, adj_thre)
    mask_dict['weight1_mask'] = get_each_mask(
        model.ginlayers[0].apply_func.mlp.linear.state_dict()['weight_mask_train'], wei_thre)
    mask_dict['weight2_mask'] = get_each_mask(
        model.ginlayers[1].apply_func.mlp.linear.state_dict()['weight_mask_train'], wei_thre)
    adj_spar = mask_dict['adj_mask'].sum() * 100 / model.edge_num
    wei_nonzero = mask_dict['weight1_mask'].sum() + mask_dict['weight2_mask'].sum()
    wei_all = mask_dict['weight1_mask'].numel() + mask_dict['weight2_mask'].numel()
    wei_spar = wei_nonzero * 100 / wei_all
    return mask_dict, adj_spar, wei_spar

def get_final_weight_mask_epoch(model, wei_percent):
    weight1 = model.ginlayers[0].apply_func.mlp.linear.weight_orig_weight.detach().cpu().flatten()
    weight2 = model.ginlayers[1].apply_func.mlp.linear.weight_orig_weight.detach().cpu().flatten()

    weight_mask_tensor = torch.cat([weight1, weight2])

    wei_y, wei_i = torch.sort(weight_mask_tensor.abs())
    wei_total = weight_mask_tensor.shape[0]

    wei_thre_index = int(wei_total * wei_percent)
    wei_thre = wei_y[wei_thre_index]

    mask_dict = {}
    mask_dict['weight1_mask'] = get_each_mask(
        model.ginlayers[0].apply_func.mlp.linear.state_dict()['weight_orig_weight'], wei_thre)
    mask_dict['weight2_mask'] = get_each_mask(
        model.ginlayers[1].apply_func.mlp.linear.state_dict()['weight_orig_weight'], wei_thre)

    return mask_dict


def print_sparsity(model):
    adj_nonzero = model.edge_num
    adj_mask_nonzero = model.adj_mask2_fixed.sum().item()
    adj_spar = adj_mask_nonzero * 100 / adj_nonzero

    weight1_total = model.ginlayers[0].apply_func.mlp.linear.weight_mask_fixed.numel()
    weight2_total = model.ginlayers[1].apply_func.mlp.linear.weight_mask_fixed.numel()
    weight_total = weight1_total + weight2_total

    weight1_nonzero = model.ginlayers[0].apply_func.mlp.linear.weight_mask_fixed.sum().item()
    weight2_nonzero = model.ginlayers[1].apply_func.mlp.linear.weight_mask_fixed.sum().item()
    weight_nonzero = weight1_nonzero + weight2_nonzero

    wei_spar = weight_nonzero * 100 / weight_total
    print("-" * 100)
    print("Sparsity: Adj:[{:.2f}%] Wei:[{:.2f}%]"
          .format(adj_spar, wei_spar))
    print("-" * 100)

    return adj_spar, wei_spar


def add_trainable_mask_noise(model, c=1e-5):
    model.adj_mask1_train.requires_grad = False
    model.ginlayers[0].apply_func.mlp.linear.weight_mask_train.requires_grad = False
    model.ginlayers[1].apply_func.mlp.linear.weight_mask_train.requires_grad = False

    rand1 = (2 * torch.rand(model.adj_mask1_train.shape) - 1) * c
    rand1 = rand1.to(model.adj_mask1_train.device)
    rand1 = rand1 * model.adj_mask1_train
    model.adj_mask1_train.add_(rand1)

    rand2 = (2 * torch.rand(model.ginlayers[0].apply_func.mlp.linear.weight_mask_train.shape) - 1) * c
    rand2 = rand2.to(model.ginlayers[0].apply_func.mlp.linear.weight_mask_train.device)
    rand2 = rand2 * model.ginlayers[0].apply_func.mlp.linear.weight_mask_train
    model.ginlayers[0].apply_func.mlp.linear.weight_mask_train.add_(rand2)

    rand3 = (2 * torch.rand(model.ginlayers[1].apply_func.mlp.linear.weight_mask_train.shape) - 1) * c
    rand3 = rand3.to(model.ginlayers[1].apply_func.mlp.linear.weight_mask_train.device)
    rand3 = rand3 * model.ginlayers[1].apply_func.mlp.linear.weight_mask_train
    model.ginlayers[1].apply_func.mlp.linear.weight_mask_train.add_(rand3)

    model.adj_mask1_train.requires_grad = True
    model.ginlayers[0].apply_func.mlp.linear.weight_mask_train.requires_grad = True
    model.ginlayers[1].apply_func.mlp.linear.weight_mask_train.requires_grad = True
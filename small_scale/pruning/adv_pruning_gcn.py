import copy
import os
os.environ['CUDA_VISIBLE_DEVICES']="0"
import torch
import torch.nn.functional as F
import numpy as np

def subgradient_update_mask(model, args):
    model.adj_mask1_train.grad.data.add_(args['s1'] * torch.sign(model.adj_mask1_train.data))
    model.gcn1.apply_mod.linear.weight_mask_train.grad.data.add_(args['s2'] * torch.sign(model.gcn1.apply_mod.linear.weight_mask_train.data))
    model.gcn2.apply_mod.linear.weight_mask_train.grad.data.add_(args['s2'] * torch.sign(model.gcn2.apply_mod.linear.weight_mask_train.data))

def get_mask_distribution(model, if_numpy=True):
    weight_mask_tensor0 = model.gcn1.apply_mod.linear.weight_mask_train.flatten()
    nonzero = torch.abs(weight_mask_tensor0) > 0
    weight_mask_tensor0 = weight_mask_tensor0[nonzero]
    weight_mask_tensor1 = model.gcn2.apply_mod.linear.weight_mask_train.flatten()
    nonzero = torch.abs(weight_mask_tensor1) > 0
    weight_mask_tensor1 = weight_mask_tensor1[nonzero]
    weight_mask_tensor = torch.cat([weight_mask_tensor0, weight_mask_tensor1])
    if if_numpy:
        return weight_mask_tensor.detach().cpu().numpy()
    else:
        return weight_mask_tensor.detach().cpu()

def get_each_mask(mask_weight_tensor, threshold):
    ones  = torch.ones_like(mask_weight_tensor)
    zeros = torch.zeros_like(mask_weight_tensor)
    mask = torch.where(mask_weight_tensor.abs() > threshold , ones, zeros)
    return mask

def get_each_mask_adv(mask_weight_tensor, threshold):
    ones  = torch.ones_like(mask_weight_tensor)
    zeros = torch.zeros_like(mask_weight_tensor)
    temp_1 = torch.where(mask_weight_tensor.abs() > 0.0, ones, zeros)
    temp_2 = torch.where(mask_weight_tensor.abs() < threshold, ones, zeros)
    mask = torch.where(temp_1 == 0, zeros, temp_2 )
    return mask

def get_final_mask_epoch(model, percent):
    wei_mask = get_mask_distribution(model, if_numpy=False)
    wei_total = wei_mask.shape[0]
    ### sort
    wei_y, wei_i = torch.sort(wei_mask.abs())
    ### get threshold
    wei_thre_index = int(wei_total * (1-percent))
    wei_thre = wei_y[wei_thre_index]
    mask_dict = {}
    mask_dict['weight1_mask'] = get_each_mask(model.gcn1.apply_mod.linear.state_dict()['weight_mask_train'], wei_thre)
    mask_dict['weight2_mask'] = get_each_mask(model.gcn2.apply_mod.linear.state_dict()['weight_mask_train'], wei_thre)
    return (mask_dict['weight1_mask'].sum()+mask_dict['weight2_mask'].sum()).int(), mask_dict

def get_final_mask_epoch_adv(model, wei_thre_index):
    wei_mask = get_mask_distribution(model, if_numpy=False)
    ### sort
    wei_y, wei_i = torch.sort(wei_mask.abs())
    ### get threshold
    wei_thre = wei_y[wei_thre_index]
    mask_dict = {}
    mask_dict['weight1_mask'] = get_each_mask_adv(model.gcn1.apply_mod.linear.state_dict()['weight_mask_train'], wei_thre)
    mask_dict['weight2_mask'] = get_each_mask_adv(model.gcn2.apply_mod.linear.state_dict()['weight_mask_train'], wei_thre)

    return mask_dict

def get_mask_distribution_adj(model, if_numpy=True):
    adj_mask_tensor = model.adj_mask1_train.flatten()
    nonzero = torch.abs(adj_mask_tensor) > 0
    adj_mask_tensor = adj_mask_tensor[nonzero]
    if if_numpy:
        return adj_mask_tensor.detach().cpu().numpy()
    else:
        return adj_mask_tensor.detach().cpu()

def get_each_mask_adj(mask_weight_tensor, threshold):
    ones = torch.ones_like(mask_weight_tensor)
    zeros = torch.zeros_like(mask_weight_tensor)
    mask = torch.where(mask_weight_tensor.abs() >= threshold, ones, zeros)
    return mask

def get_each_mask_adv_adj(mask_weight_tensor, threshold):
    # origin
    ones  = torch.ones_like(mask_weight_tensor)
    zeros = torch.zeros_like(mask_weight_tensor)
    temp_1 = torch.where(mask_weight_tensor.abs() <= threshold , ones, zeros)
    temp_2 = torch.where(mask_weight_tensor.abs() > 0.0, ones, zeros)
    mask = torch.where(temp_1 == temp_2, ones, zeros)
    return mask

def get_final_mask_epoch_adj(model_adv, recover_rate):
    adj_mask = get_mask_distribution_adj(model_adv, if_numpy=False)
    adj_total = adj_mask.shape[0]
    adj_y, adj_i = torch.sort(adj_mask.abs())
    adj_thre_index = int(adj_total * (1-recover_rate))
    adj_thre = adj_y[adj_thre_index]
    mask_dict = {}
    ori_adj_mask = model_adv.adj_mask1_train.detach().cpu()
    mask_dict['adj_mask'] = get_each_mask_adj(ori_adj_mask, adj_thre)
    return mask_dict['adj_mask'].sum().int(), mask_dict

def get_final_mask_epoch_adv_adj(model_origin, adj_thre_index):
    adj_mask = get_mask_distribution_adj(model_origin, if_numpy=False)
    adj_y, adj_i = torch.sort(adj_mask.abs())
    mask_dict = {}
    ori_adj_mask = model_origin.adj_mask1_train.detach().cpu()
    adj_thre = adj_y[adj_thre_index]
    mask_dict['adj_mask'] = get_each_mask_adv_adj(ori_adj_mask, adj_thre)
    return mask_dict

def get_final_mask_round_origin(model, best_epoch_mask_adv, best_epoch_mask_origin):
    ones_0 = torch.ones_like(model.gcn1.apply_mod.linear.weight_mask_train)
    zeros_0 = torch.zeros_like(model.gcn1.apply_mod.linear.weight_mask_train)
    mask_0 = torch.where(model.gcn1.apply_mod.linear.weight_mask_train != 0, ones_0, zeros_0)
    temp1_0 = torch.where(best_epoch_mask_adv['weight1_mask'] == 1, ones_0, mask_0)
    temp2_0 = torch.where(best_epoch_mask_origin['weight1_mask'] == 1, zeros_0, temp1_0)
    ones_1 = torch.ones_like(model.gcn2.apply_mod.linear.weight_mask_train)
    zeros_1 = torch.zeros_like(model.gcn2.apply_mod.linear.weight_mask_train)
    mask_1 = torch.where(model.gcn2.apply_mod.linear.weight_mask_train != 0, ones_1, zeros_1)
    temp1_1 = torch.where(best_epoch_mask_adv['weight2_mask'] == 1, ones_1, mask_1)
    temp2_1 = torch.where(best_epoch_mask_origin['weight2_mask'] == 1, zeros_1, temp1_1)
    mask_dict={}
    mask_dict['weight1_mask']=temp2_0
    mask_dict['weight2_mask']=temp2_1
    return mask_dict

def get_final_mask_round_adv(model, best_epoch_mask_adv, best_epoch_mask_origin):
    ones_0 = torch.ones_like(model.gcn1.apply_mod.linear.weight_mask_train)
    zeros_0 = torch.zeros_like(model.gcn1.apply_mod.linear.weight_mask_train)
    mask_0 = torch.where(model.gcn1.apply_mod.linear.weight_mask_train != 0, ones_0, zeros_0)
    temp1_0 = torch.where(best_epoch_mask_adv['weight1_mask'] == 1, zeros_0, mask_0)
    temp2_0 = torch.where(best_epoch_mask_origin['weight1_mask'] == 1, ones_0, temp1_0)
    ones_1 = torch.ones_like(model.gcn2.apply_mod.linear.weight_mask_train)
    zeros_1 = torch.zeros_like(model.gcn2.apply_mod.linear.weight_mask_train)
    mask_1 = torch.where(model.gcn2.apply_mod.linear.weight_mask_train != 0, ones_1, zeros_1)
    temp1_1 = torch.where(best_epoch_mask_adv['weight2_mask'] == 1, zeros_1, mask_1)
    temp2_1 = torch.where(best_epoch_mask_origin['weight2_mask'] == 1, ones_1, temp1_1)
    mask_dict = {}
    mask_dict['weight1_mask'] = temp2_0
    mask_dict['weight2_mask'] = temp2_1
    return mask_dict

def get_final_mask_round_origin_adj(model, best_epoch_mask_adv, best_epoch_mask_origin):
    ones = torch.ones_like(model.state_dict()['adj_mask1_train'])
    zeros = torch.zeros_like(model.state_dict()['adj_mask1_train'])
    mask = torch.where(model.state_dict()['adj_mask1_train'] != 0.0, ones, zeros)
    temp1 = torch.where(best_epoch_mask_adv['adj_mask'] == 1, ones.cpu(), mask.cpu())
    temp2 = torch.where(best_epoch_mask_origin['adj_mask'] == 1, zeros.cpu(), temp1.cpu())
    mask_dict={}
    mask_dict['adj_mask']=temp2
    return mask_dict

def get_final_mask_round_adv_adj(model, best_epoch_mask_adv, best_epoch_mask_origin):
    ones = torch.ones_like(model.adj_mask1_train)
    zeros = torch.zeros_like(model.adj_mask1_train)
    mask = torch.where(model.adj_mask1_train != 0, ones, zeros)
    temp1 = torch.where(best_epoch_mask_adv['adj_mask'] == 1, zeros.cpu(), mask.cpu())
    temp2 = torch.where(best_epoch_mask_origin['adj_mask'] == 1, ones.cpu(), temp1.cpu())
    mask_dict = {}
    mask_dict['adj_mask'] = temp2
    return mask_dict

def get_final_mask_round_origin_both(model, best_epoch_mask_adv, best_epoch_mask_origin, best_epoch_mask_adv_adj, best_epoch_mask_origin_adj):
    ones_0 = torch.ones_like(model.gcn1.apply_mod.linear.weight_mask_train)
    zeros_0 = torch.zeros_like(model.gcn1.apply_mod.linear.weight_mask_train)
    mask_0 = torch.where(model.gcn1.apply_mod.linear.weight_mask_train != 0, ones_0, zeros_0)
    temp1_0 = torch.where(best_epoch_mask_adv['weight1_mask'] == 1, ones_0, mask_0)
    temp2_0 = torch.where(best_epoch_mask_origin['weight1_mask'] == 1, zeros_0, temp1_0)
    ones_1 = torch.ones_like(model.gcn2.apply_mod.linear.weight_mask_train)
    zeros_1 = torch.zeros_like(model.gcn2.apply_mod.linear.weight_mask_train)
    mask_1 = torch.where(model.gcn2.apply_mod.linear.weight_mask_train != 0, ones_1, zeros_1)
    temp1_1 = torch.where(best_epoch_mask_adv['weight2_mask'] == 1, ones_1, mask_1)
    temp2_1 = torch.where(best_epoch_mask_origin['weight2_mask'] == 1, zeros_1, temp1_1)
    ones = torch.ones_like(model.state_dict()['adj_mask1_train'])
    zeros = torch.zeros_like(model.state_dict()['adj_mask1_train'])
    mask = torch.where(model.state_dict()['adj_mask1_train'] != 0.0, ones, zeros)
    temp1 = torch.where(best_epoch_mask_adv_adj['adj_mask'] == 1, ones.cpu(), mask.cpu())
    temp2 = torch.where(best_epoch_mask_origin_adj['adj_mask'] == 1, zeros.cpu(), temp1.cpu())
    mask_dict={}
    mask_dict['adj_mask']=temp2
    mask_dict['weight1_mask']=temp2_0
    mask_dict['weight2_mask']=temp2_1
    return mask_dict

def get_final_mask_round_adv_both(model, best_epoch_mask_adv, best_epoch_mask_origin, best_epoch_mask_adv_adj, best_epoch_mask_origin_adj):
    ones_0 = torch.ones_like(model.gcn1.apply_mod.linear.weight_mask_train)
    zeros_0 = torch.zeros_like(model.gcn1.apply_mod.linear.weight_mask_train)
    mask_0 = torch.where(model.gcn1.apply_mod.linear.weight_mask_train != 0, ones_0, zeros_0)
    temp1_0 = torch.where(best_epoch_mask_adv['weight1_mask'] == 1, zeros_0, mask_0)
    temp2_0 = torch.where(best_epoch_mask_origin['weight1_mask'] == 1, ones_0, temp1_0)
    ones_1 = torch.ones_like(model.gcn2.apply_mod.linear.weight_mask_train)
    zeros_1 = torch.zeros_like(model.gcn2.apply_mod.linear.weight_mask_train)
    mask_1 = torch.where(model.gcn2.apply_mod.linear.weight_mask_train != 0, ones_1, zeros_1)
    temp1_1 = torch.where(best_epoch_mask_adv['weight2_mask'] == 1, zeros_1, mask_1)
    temp2_1 = torch.where(best_epoch_mask_origin['weight2_mask'] == 1, ones_1, temp1_1)
    ones = torch.ones_like(model.adj_mask1_train)
    zeros = torch.zeros_like(model.adj_mask1_train)
    mask = torch.where(model.adj_mask1_train != 0, ones, zeros)
    temp1 = torch.where(best_epoch_mask_adv_adj['adj_mask'] == 1, zeros.cpu(), mask.cpu())
    temp2 = torch.where(best_epoch_mask_origin_adj['adj_mask'] == 1, ones.cpu(), temp1.cpu())
    mask_dict = {}
    mask_dict['adj_mask']=temp2
    mask_dict['weight1_mask'] = temp2_0
    mask_dict['weight2_mask'] = temp2_1
    return mask_dict

def get_new_mask_high(logits, ratio):
    prob = F.softmax(logits)
    p = np.array(prob.cpu())
    p /= p.sum()
    indices = np.random.choice(len(logits), size=ratio, p=p)
    return indices

def get_new_mask_low(logits, ratio):
    prob = F.softmax(logits)
    prob = 1-prob
    p = np.array(prob.cpu())
    p /= p.sum()
    indices = np.random.choice(len(logits), size=ratio, p=p)
    return indices

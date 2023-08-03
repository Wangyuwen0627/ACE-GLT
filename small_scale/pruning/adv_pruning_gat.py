import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch

def subgradient_update_mask(model, args):
    model.adj_mask1_train.grad.data.add_(args['s1'] * torch.sign(model.adj_mask1_train.data))
    for layer in range(1):
        for head in range(8):
            model.layers[layer].heads[head].fc.weight_mask_train.grad.data.add_(
                args['s2'] * torch.sign(model.layers[layer].heads[head].fc.weight_mask_train.data))
            model.layers[layer].heads[head].attn_fc.weight_mask_train.grad.data.add_(
                args['s2'] * torch.sign(model.layers[layer].heads[head].attn_fc.weight_mask_train.data))
            if layer == 1: break


def get_mask_distribution(model):
    weight_mask_vector = torch.tensor([]).to(torch.device("cuda:0"))
    for layer in range(1):
        for head in range(8):
            weight_mask1 = model.layers[layer].heads[head].fc.weight_mask_train.flatten()
            nonzero = torch.abs(weight_mask1) > 0
            weight_mask1 = weight_mask1[nonzero]
            weight_mask2 = model.layers[layer].heads[head].attn_fc.weight_mask_train.flatten()
            nonzero = torch.abs(weight_mask2) > 0
            weight_mask2 = weight_mask2[nonzero]
            weight_mask_vector = torch.cat((weight_mask_vector, weight_mask1))
            weight_mask_vector = torch.cat((weight_mask_vector, weight_mask2))
            if layer == 1: break
    return weight_mask_vector.detach().cpu()


def get_each_mask(mask_weight_tensor, threshold):
    # origin
    ones = torch.ones_like(mask_weight_tensor)
    zeros = torch.zeros_like(mask_weight_tensor)
    mask = torch.where(mask_weight_tensor.abs() > threshold, ones, zeros)
    return mask


def get_each_mask_adv(mask_weight_tensor, threshold):
    # origin
    ones = torch.ones_like(mask_weight_tensor)
    zeros = torch.zeros_like(mask_weight_tensor)
    temp_1 = torch.where(mask_weight_tensor.abs() > 0.0, ones, zeros)
    temp_2 = torch.where(mask_weight_tensor.abs() < threshold, ones, zeros)
    mask = torch.where(temp_1 == 0, zeros, temp_2)
    return mask


##### pruning remain mask percent #######
def get_final_mask_epoch(model, percent):
    wei_mask = get_mask_distribution(model)
    wei_total = wei_mask.shape[0]
    ### sort
    wei_y, wei_i = torch.sort(wei_mask.abs())
    ### get threshold
    wei_thre_index = int(wei_total * (1 - percent))
    wei_thre = wei_y[wei_thre_index]
    mask_dict = {}
    num = 0
    for layer in range(1):
        for head in range(8):
            key_train1 = 'layers.{}.heads.{}.fc.weight_mask_train'.format(layer, head)
            key_fixed1 = 'layers.{}.heads.{}.fc.weight_mask_fixed'.format(layer, head)
            key_train2 = 'layers.{}.heads.{}.attn_fc.weight_mask_train'.format(layer, head)
            key_fixed2 = 'layers.{}.heads.{}.attn_fc.weight_mask_fixed'.format(layer, head)
            mask_dict[key_train1] = get_each_mask(model.layers[layer].heads[head].fc.state_dict()['weight_mask_train'],
                                                  wei_thre)
            mask_dict[key_train2] = get_each_mask(
                model.layers[layer].heads[head].attn_fc.state_dict()['weight_mask_train'], wei_thre)
            num = num + mask_dict[key_train1].sum().int() + mask_dict[key_train2].sum().int()
            if layer == 1: break

    return num, mask_dict


##### pruning remain mask percent #######
def get_final_mask_epoch_adv(model, wei_thre_index):
    wei_mask = get_mask_distribution(model)
    ### sort
    wei_y, wei_i = torch.sort(wei_mask.abs())
    ### get threshold
    wei_thre = wei_y[wei_thre_index]
    mask_dict = {}
    num = 0
    for layer in range(1):
        for head in range(8):
            key_train1 = 'layers.{}.heads.{}.fc.weight_mask_train'.format(layer, head)
            key_train2 = 'layers.{}.heads.{}.attn_fc.weight_mask_train'.format(layer, head)
            mask_dict[key_train1] = get_each_mask_adv(
                model.layers[layer].heads[head].fc.state_dict()['weight_mask_train'], wei_thre)
            mask_dict[key_train2] = get_each_mask_adv(
                model.layers[layer].heads[head].attn_fc.state_dict()['weight_mask_train'], wei_thre)
            if layer == 1: break

    return mask_dict


def get_final_mask_round_origin(model, best_epoch_mask_adv, best_epoch_mask_origin):
    mask_dict = {}
    for layer in range(1):
        for head in range(8):
            key_train1 = 'layers.{}.heads.{}.fc.weight_mask_train'.format(layer, head)
            key_train2 = 'layers.{}.heads.{}.attn_fc.weight_mask_train'.format(layer, head)
            ones_0 = torch.ones_like(model.layers[layer].heads[head].fc.state_dict()['weight_mask_train'])
            zeros_0 = torch.zeros_like(model.layers[layer].heads[head].fc.state_dict()['weight_mask_train'])
            mask_0 = torch.where(model.layers[layer].heads[head].fc.state_dict()['weight_mask_train'] != 0, ones_0,
                                 zeros_0)
            temp1_0 = torch.where(best_epoch_mask_adv[key_train1] == 1, ones_0, mask_0)
            temp2_0 = torch.where(best_epoch_mask_origin[key_train1] == 1, zeros_0, temp1_0)
            ones_1 = torch.ones_like(model.layers[layer].heads[head].attn_fc.state_dict()['weight_mask_train'])
            zeros_1 = torch.zeros_like(model.layers[layer].heads[head].attn_fc.state_dict()['weight_mask_train'])
            mask_1 = torch.where(model.layers[layer].heads[head].attn_fc.state_dict()['weight_mask_train'] != 0, ones_1,
                                 zeros_1)
            temp1_1 = torch.where(best_epoch_mask_adv[key_train2] == 1, ones_1, mask_1)
            temp2_1 = torch.where(best_epoch_mask_origin[key_train2] == 1, zeros_1, temp1_1)
            mask_dict[key_train1] = temp2_0
            mask_dict[key_train2] = temp2_1
            if layer == 1: break
    return mask_dict


def get_final_mask_round_adv(model, best_epoch_mask_adv, best_epoch_mask_origin):
    mask_dict = {}
    for layer in range(1):
        for head in range(8):
            key_train1 = 'layers.{}.heads.{}.fc.weight_mask_train'.format(layer, head)
            key_train2 = 'layers.{}.heads.{}.attn_fc.weight_mask_train'.format(layer, head)
            ones_0 = torch.ones_like(model.layers[layer].heads[head].fc.state_dict()['weight_mask_train'])
            zeros_0 = torch.zeros_like(model.layers[layer].heads[head].fc.state_dict()['weight_mask_train'])
            mask_0 = torch.where(model.layers[layer].heads[head].fc.state_dict()['weight_mask_train'] != 0, ones_0,
                                 zeros_0)
            temp1_0 = torch.where(best_epoch_mask_adv[key_train1] == 1, zeros_0, mask_0)
            temp2_0 = torch.where(best_epoch_mask_origin[key_train1] == 1, ones_0, temp1_0)
            ones_1 = torch.ones_like(model.layers[layer].heads[head].attn_fc.state_dict()['weight_mask_train'])
            zeros_1 = torch.zeros_like(model.layers[layer].heads[head].attn_fc.state_dict()['weight_mask_train'])
            mask_1 = torch.where(model.layers[layer].heads[head].attn_fc.state_dict()['weight_mask_train'] != 0, ones_1,
                                 zeros_1)
            temp1_1 = torch.where(best_epoch_mask_adv[key_train2] == 1, zeros_1, mask_1)
            temp2_1 = torch.where(best_epoch_mask_origin[key_train2] == 1, ones_1, temp1_1)
            mask_dict[key_train1] = temp2_0
            mask_dict[key_train2] = temp2_1
            if layer == 1: break
    return mask_dict


def get_mask_distribution_adj(model, if_numpy=True):
    adj_mask_tensor = model.adj_mask1_train.flatten()
    nonzero = torch.abs(adj_mask_tensor) > 0
    adj_mask_tensor = adj_mask_tensor[nonzero]  # 13264 - 2708
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
    ones = torch.ones_like(mask_weight_tensor)
    zeros = torch.zeros_like(mask_weight_tensor)
    temp_1 = torch.where(mask_weight_tensor.abs() <= threshold, ones, zeros)
    temp_2 = torch.where(mask_weight_tensor.abs() > 0.0, ones, zeros)
    mask = torch.where(temp_1 == temp_2, ones, zeros)
    return mask


def get_final_mask_epoch_adj(model_adv, recover_rate):
    adj_mask = get_mask_distribution_adj(model_adv, if_numpy=False)
    adj_total = adj_mask.shape[0]
    adj_y, adj_i = torch.sort(adj_mask.abs())
    adj_thre_index = int(adj_total * (1 - recover_rate))
    adj_thre = adj_y[adj_thre_index]
    mask_dict = {}
    ori_adj_mask = model_adv.adj_mask1_train.detach().cpu()
    # ori_adj_mask.add_((2 * torch.rand(ori_adj_mask.shape) - 1) * 1e-5)
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


def get_final_mask_round_origin_adj(model, best_epoch_mask_adv, best_epoch_mask_origin):
    ones = torch.ones_like(model.state_dict()['adj_mask1_train'])
    zeros = torch.zeros_like(model.state_dict()['adj_mask1_train'])
    mask = torch.where(model.state_dict()['adj_mask1_train'] != 0.0, ones, zeros)
    temp1 = torch.where(best_epoch_mask_adv['adj_mask'] == 1, ones.cpu(), mask.cpu())
    temp2 = torch.where(best_epoch_mask_origin['adj_mask'] == 1, zeros.cpu(), temp1.cpu())
    mask_dict = {}
    mask_dict['adj_mask'] = temp2
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


def get_sparsity(model):
    adj_nonzero = model.edge_num
    adj_mask_nonzero = model.adj_mask2_fixed.sum().item()
    adj_spar = adj_mask_nonzero * 100 / adj_nonzero
    weight_total = 0
    weight_nonzero = 0
    for layer in range(1):
        for head in range(8):
            weight_total += model.layers[layer].heads[head].fc.weight_mask_fixed.numel()
            weight_total += model.layers[layer].heads[head].attn_fc.weight_mask_fixed.numel()
            weight_nonzero += model.layers[layer].heads[head].fc.weight_mask_fixed.sum().item()
            weight_nonzero += model.layers[layer].heads[head].attn_fc.weight_mask_fixed.sum().item()
            if layer == 1: break
    wei_spar = 100-weight_nonzero * 100 / weight_total
    return adj_spar, wei_spar

def print_sparsity(model):
    adj_nonzero = model.edge_num
    adj_mask_nonzero = model.adj_mask2_fixed.sum().item()
    adj_spar = adj_mask_nonzero * 100 / adj_nonzero

    weight_total = 0
    weight_nonzero = 0
    for layer in range(1):
        for head in range(8):
            weight_total += model.layers[layer].heads[head].fc.weight_mask_fixed.numel()
            weight_total += model.layers[layer].heads[head].attn_fc.weight_mask_fixed.numel()
            weight_nonzero += model.layers[layer].heads[head].fc.weight_mask_fixed.sum().item()
            weight_nonzero += model.layers[layer].heads[head].attn_fc.weight_mask_fixed.sum().item()
            if layer == 1: break
    wei_spar = 100 - weight_nonzero * 100 / weight_total
    print("-" * 100)
    print("Sparsity: Adj:[{:.2f}%] Wei:[{:.2f}%]".format(adj_spar, wei_spar))
    print("-" * 100)
    return adj_spar, wei_spar
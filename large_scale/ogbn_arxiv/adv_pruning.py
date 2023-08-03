import torch

def get_soft_mask_distribution(model):

    adj_mask_vector = model.edge_mask1_train.flatten()
    nonzero = torch.abs(adj_mask_vector) > 0
    adj_mask_vector = adj_mask_vector[nonzero]

    weight_mask_vector = torch.tensor([]).to(torch.device("cuda:0"))
    for i in range(28):
        weight_mask = model.gcns[i].mlp[0].weight_mask_train.flatten()
        nonzero = torch.abs(weight_mask) > 0
        weight_mask = weight_mask[nonzero]
        weight_mask_vector = torch.cat((weight_mask_vector, weight_mask))
    
    return adj_mask_vector.detach().cpu(), weight_mask_vector.detach().cpu()


def get_each_mask(mask_weight_tensor, threshold):
    
    ones  = torch.ones_like(mask_weight_tensor)
    zeros = torch.zeros_like(mask_weight_tensor) 
    mask = torch.where(mask_weight_tensor.abs() > threshold, ones, zeros)
    return mask

def get_each_mask_adv(mask_weight_tensor, threshold):
    # origin
    ones  = torch.ones_like(mask_weight_tensor)
    zeros = torch.zeros_like(mask_weight_tensor)
    temp_1 = torch.where(mask_weight_tensor.abs() > 0.0, ones, zeros)
    temp_2 = torch.where(mask_weight_tensor.abs() < threshold, ones, zeros)
    mask = torch.where(temp_1 == 0, zeros, temp_2)
    return mask

##### pruning remain mask percent #######
def get_final_mask_epoch(model, percent, num_layer=28):
    wei_mask = get_soft_mask_distribution(model)
    wei_total = wei_mask.shape[0]
    ### sort
    wei_y, wei_i = torch.sort(wei_mask.abs())
    ### get threshold
    wei_thre_index = int(wei_total * (1 - percent))
    wei_thre = wei_y[wei_thre_index]
    mask_dict = {}
    num = 0
    for i in range(num_layer):
        key_train = 'gcns.{}.mlp.0.weight_mask_train'.format(i)
        key_fixed = 'gcns.{}.mlp.0.weight_mask_fixed'.format(i)
        mask_dict[key_train] = get_each_mask(model.gcns[i].mlp[0].state_dict()['weight_mask_train'], wei_thre)
        mask_dict[key_fixed] = mask_dict[key_train]
        num = num + mask_dict[key_train].sum().int()
    return num, mask_dict

##### pruning remain mask percent #######
def get_final_mask_epoch_adv(model, wei_thre_index, num_layer=28):
    wei_mask = get_soft_mask_distribution(model)
    ### sort
    wei_y, wei_i = torch.sort(wei_mask.abs())
    ### get threshold
    wei_thre = wei_y[wei_thre_index]
    mask_dict = {}
    for i in range(num_layer):
        key_train = 'gcns.{}.mlp.0.weight_mask_train'.format(i)
        key_fixed = 'gcns.{}.mlp.0.weight_mask_fixed'.format(i)
        mask_dict[key_train] = get_each_mask_adv(model.gcns[i].mlp[0].state_dict()['weight_mask_train'], wei_thre)
        mask_dict[key_fixed] = mask_dict[key_train]
    return mask_dict

def get_final_mask_round_origin(model, best_epoch_mask_adv, best_epoch_mask_origin, num_layer=28):
    mask_dict={}
    for i in range(num_layer):
        key_train = 'gcns.{}.mlp.0.weight_mask_train'.format(i)
        key_fixed = 'gcns.{}.mlp.0.weight_mask_fixed'.format(i)
        ones_0 = torch.ones_like(model.gcns[i].mlp[0].state_dict()['weight_mask_train'])
        zeros_0 = torch.zeros_like(model.gcns[i].mlp[0].state_dict()['weight_mask_train'])
        mask_0 = torch.where(model.gcns[i].mlp[0].state_dict()['weight_mask_train'] != 0, ones_0, zeros_0)
        temp1_0 = torch.where(best_epoch_mask_adv[key_train] == 1, ones_0, mask_0)
        temp2_0 = torch.where(best_epoch_mask_origin[key_train] == 1, zeros_0, temp1_0)
        mask_dict[key_train] = temp2_0
        mask_dict[key_fixed] = temp2_0
    return mask_dict

def get_final_mask_round_adv(model, best_epoch_mask_adv, best_epoch_mask_origin, num_layer=28):
    mask_dict = {}
    for i in range(num_layer):
        key_train = 'gcns.{}.mlp.0.weight_mask_train'.format(i)
        key_fixed = 'gcns.{}.mlp.0.weight_mask_fixed'.format(i)
        ones_0 = torch.ones_like(model.gcns[i].mlp[0].state_dict()['weight_mask_train'])
        zeros_0 = torch.zeros_like(model.gcns[i].mlp[0].state_dict()['weight_mask_train'])
        mask_0 = torch.where(model.gcns[i].mlp[0].state_dict()['weight_mask_train'] != 0, ones_0, zeros_0)
        temp1_0 = torch.where(best_epoch_mask_adv[key_train] == 1, zeros_0, mask_0)
        temp2_0 = torch.where(best_epoch_mask_origin[key_train] == 1, ones_0, temp1_0)
        mask_dict[key_train] = temp2_0
        mask_dict[key_fixed] = temp2_0
    return mask_dict
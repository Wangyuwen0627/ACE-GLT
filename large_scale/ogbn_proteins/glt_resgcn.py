import torch
import torch.optim as optim
from dataset import OGBNDataset
from model import DeeperGCN
from args import ArgsInit
import time
from ogb.nodeproppred import Evaluator
import train
import adv_pruning
import pruning
import copy
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def main_fixed_mask(args, imp_num, final_state_dict=None, resume_train_ckpt=None):

    device = torch.device("cuda:" + str(args.device))
    dataset = OGBNDataset(dataset_name=args.dataset)
    nf_path = dataset.extract_node_features(args.aggr)

    args.num_tasks = dataset.num_tasks
    args.nf_path = nf_path

    evaluator = Evaluator(args.dataset)
    criterion = torch.nn.BCEWithLogitsLoss()

    valid_data_list = []
    for i in range(args.num_evals):

        parts = dataset.random_partition_graph(dataset.total_no_of_nodes, cluster_number=args.valid_cluster_number)
        valid_data = dataset.generate_sub_graphs(parts, cluster_number=args.valid_cluster_number)
        valid_data_list.append(valid_data)

    print("-" * 120)
    model = DeeperGCN(args).to(device)
    pruning.add_mask(model)

    if final_state_dict is not None:
        
        pruning.retrain_operation(dataset, model, final_state_dict)
        adj_spar, wei_spar = pruning.print_sparsity(dataset, model)

    for name, param in model.named_parameters():
        if 'mask' in name:
            param.requires_grad = False

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    results = {'highest_valid': 0, 'final_train': 0, 'final_test': 0, 'highest_train': 0, 'epoch':0}
    results['adj_spar'] = adj_spar
    results['wei_spar'] = wei_spar
    
    start_epoch = 1
    if resume_train_ckpt:
        dataset.adj = resume_train_ckpt['adj']
        start_epoch = resume_train_ckpt['epoch']
        rewind_weight_mask = resume_train_ckpt['rewind_weight_mask']
        ori_model_dict = model.state_dict()
        over_lap = {k : v for k, v in resume_train_ckpt['model_state_dict'].items() if k in ori_model_dict.keys()}
        ori_model_dict.update(over_lap)
        model.load_state_dict(ori_model_dict)
        print("Resume at IMP:[{}] epoch:[{}] len:[{}/{}]!".format(imp_num, resume_train_ckpt['epoch'], len(over_lap.keys()), len(ori_model_dict.keys())))
        optimizer.load_state_dict(resume_train_ckpt['optimizer_state_dict'])
        adj_spar, wei_spar = pruning.print_sparsity(dataset, model)
    
    for epoch in range(start_epoch, args.fix_epochs + 1):
        # do random partition every epoch
        t0 = time.time()
        train_parts = dataset.random_partition_graph(dataset.total_no_of_nodes, cluster_number=args.cluster_number)
        data = dataset.generate_sub_graphs(train_parts, cluster_number=args.cluster_number, ifmask=True)
        epoch_loss = train.train_fixed(data, dataset, model, optimizer, criterion, device, args)
        result = train.multi_evaluate(valid_data_list, dataset, model, evaluator, device)

        train_result = result['train']['rocauc']
        valid_result = result['valid']['rocauc']
        test_result = result['test']['rocauc']

        if valid_result > results['highest_valid']:
            results['highest_valid'] = valid_result
            results['final_train'] = train_result
            results['final_test'] = test_result
            results['epoch'] = epoch
            final_state_dict = pruning.save_all(dataset,
                                                model,
                                                None,
                                                optimizer,
                                                imp_num,
                                                epoch,
                                                args.model_save_path, 
                                                'IMP{}_fixed_ckpt'.format(imp_num))
        epoch_time = (time.time() - t0) / 60
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' | ' +
              'IMP:[{}] (FIX Mask) Epoch[{}/{}] LOSS[{:.4f}] Train[{:.2f}] Valid[{:.2f}] Test[{:.2f}] | Update Test[{:.2f}] at epoch[{}] | Adj[{:.2f}%] Wei[{:.2f}%] Time[{:.2f}min]'
              .format(imp_num, epoch, args.fix_epochs, epoch_loss, train_result * 100,
                                                               valid_result * 100,
                                                               test_result * 100,
                                                               results['final_test'] * 100,
                                                               results['epoch'],
                                                               results['adj_spar'] * 100,
                                                               results['wei_spar'] * 100,
                                                               epoch_time))
    print("=" * 120)
    print("INFO final: IMP:[{}], Train:[{:.2f}]  Best Val:[{:.2f}] at epoch:[{}] | Final Test Acc:[{:.2f}] | Adj:[{:.2f}%] Wei:[{:.2f}%]"
        .format(imp_num,    results['final_train'] * 100,
                            results['highest_valid'] * 100,
                            results['epoch'],
                            results['final_test'] * 100,
                            results['adj_spar'] * 100,
                            results['wei_spar'] * 100))
    print("=" * 120)

def main_get_mask(args, imp_num, resume_train_ckpt=None):

    device = torch.device("cuda:" + str(args.device))
    dataset = OGBNDataset(dataset_name=args.dataset)
    # extract initial node features
    nf_path = dataset.extract_node_features(args.aggr)

    args.num_tasks = dataset.num_tasks
    args.nf_path = nf_path

    evaluator = Evaluator(args.dataset)
    criterion = torch.nn.BCEWithLogitsLoss()

    valid_data_list = []
    for i in range(args.num_evals):

        parts = dataset.random_partition_graph(dataset.total_no_of_nodes, cluster_number=args.valid_cluster_number)
        valid_data = dataset.generate_sub_graphs(parts, cluster_number=args.valid_cluster_number)
        valid_data_list.append(valid_data)

    print("-" * 120)
    model = DeeperGCN(args).to(device)
    pruning.add_mask(model)
    pruning.add_trainable_mask_noise(model, c=1e-5)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    results = {'highest_valid': 0, 'final_train': 0, 'final_test': 0, 'highest_train': 0, 'epoch':0}
    
    start_epoch = 1
    if resume_train_ckpt:

        dataset.adj = resume_train_ckpt['adj']
        start_epoch = resume_train_ckpt['epoch']
        rewind_weight_mask = resume_train_ckpt['rewind_weight_mask']
        ori_model_dict = model.state_dict()
        over_lap = {k : v for k, v in resume_train_ckpt['model_state_dict'].items() if k in ori_model_dict.keys()}
        ori_model_dict.update(over_lap)
        model.load_state_dict(ori_model_dict)
        print("Resume at IMP:[{}] epoch:[{}] len:[{}/{}]!".format(imp_num, resume_train_ckpt['epoch'], len(over_lap.keys()), len(ori_model_dict.keys())))
        optimizer.load_state_dict(resume_train_ckpt['optimizer_state_dict'])
        pruning.print_sparsity(dataset, model)
    else:
        rewind_weight_mask = copy.deepcopy(model.state_dict())

    for epoch in range(start_epoch, args.mask_epochs + 1):
        # do random partition every epoch
        t0 = time.time()
        train_parts = dataset.random_partition_graph(dataset.total_no_of_nodes, cluster_number=args.cluster_number)
        data = dataset.generate_sub_graphs(train_parts, cluster_number=args.cluster_number, ifmask=True)
        epoch_loss, adj_spar, wei_spar = train.train_mask(epoch, data, dataset, model, optimizer, criterion, device, args)
        result = train.multi_evaluate(valid_data_list, dataset, model, evaluator, device)

        train_result = result['train']['rocauc']
        valid_result = result['valid']['rocauc']
        test_result = result['test']['rocauc']
        
        if valid_result > results['highest_valid']:
            results['highest_valid'] = valid_result
            results['final_train'] = train_result
            results['final_test'] = test_result
            results['epoch'] = epoch
            final_state_dict = pruning.save_all(dataset,
                                                model,
                                                rewind_weight_mask,
                                                optimizer,
                                                imp_num,
                                                epoch,
                                                args.model_save_path, 
                                                'IMP{}_train_ckpt'.format(imp_num))
        epoch_time = (time.time() - t0) / 60
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' | ' +
              'IMP:[{}] (GET Mask) Epoch[{}/{}] LOSS[{:.4f}] Train[{:.2f}] Valid[{:.2f}] Test[{:.2f}] | Update Test[{:.2f}] at epoch[{}] | Adj[{:.3f}%] Wei[{:.3f}%] Time:[{:.2f}min]'
              .format(imp_num, epoch, args.mask_epochs, epoch_loss, train_result * 100,
                                                    valid_result * 100,
                                                    test_result * 100,
                                                    results['final_test'] * 100,
                                                    results['epoch'], 
                                                    adj_spar * 100, 
                                                    wei_spar * 100,
                                                    epoch_time))
    print('-' * 100)
    print("INFO : IMP:[{}] (GET MASK) Final Result Train:[{:.2f}] Valid:[{:.2f}] Test:[{:.2f}] | Adj:[{:.3f}%] Wei:[{:.3f}%]"
        .format(imp_num, results['final_train'] * 100,
                         results['highest_valid'] * 100,
                         results['final_test'] * 100,
                         adj_spar * 100, 
                         wei_spar * 100))
    print('-' * 100)
    return final_state_dict

def reverse_weight_mask(rewind_weight_mask):
    ones = torch.ones_like(rewind_weight_mask)
    zeros = torch.zeros_like(rewind_weight_mask)
    mask = torch.where(rewind_weight_mask == 0.0, ones, zeros)
    return mask


def main_adv_train(args, imp_num, rewind_weight, resume_train_ckpt=None, num_layers = 28):
    device = torch.device("cuda:" + str(args.device))
    dataset = OGBNDataset(dataset_name=args.dataset)
    # extract initial node features
    nf_path = dataset.extract_node_features(args.aggr)

    args.num_tasks = dataset.num_tasks
    args.nf_path = nf_path

    evaluator = Evaluator(args.dataset)
    criterion = torch.nn.BCEWithLogitsLoss()

    valid_data_list = []
    for i in range(args.num_evals):
        parts = dataset.random_partition_graph(dataset.total_no_of_nodes, cluster_number=args.valid_cluster_number)
        valid_data = dataset.generate_sub_graphs(parts, cluster_number=args.valid_cluster_number)
        valid_data_list.append(valid_data)
    rewind_weight_mask_copy = copy.deepcopy(rewind_weight)
    num_layers = args.num_layers
    for i in range(num_layers):
        key_train = 'gcns.{}.mlp.0.weight_mask_train'.format(i)
        key_fixed = 'gcns.{}.mlp.0.weight_mask_fixed'.format(i)
        rewind_weight_mask_copy[key_train] = rewind_weight[key_train]
        rewind_weight_mask_copy[key_fixed] = rewind_weight[key_fixed]

    start_epoch = 1
    best_val_history = {'highest_valid': 0, 'round': 0, 'epoch': 0, 'final_test': 0, 'final_train': 0}
    best_dict_history = {}
    rounds = 6
    # 这里储存了彩票模型的初始化权重
    origin_weight_temp = copy.deepcopy(rewind_weight)
    # 反彩票模型的初始化权重也是相同的
    adv_weight_temp = copy.deepcopy(rewind_weight_mask_copy)
    model_origin = DeeperGCN(args).to(device)
    model_adv = DeeperGCN(args).to(device)
    pruning.add_mask(model_origin)
    pruning.add_mask(model_adv)
    pruning.add_trainable_mask_noise(model_origin, c=1e-5)
    pruning.add_trainable_mask_noise(model_adv, c=1e-5)
    rewind_weight_mask_copy = copy.deepcopy(rewind_weight)
    for i in range(num_layers):
        key_train = 'gcns.{}.mlp.0.weight_mask_train'.format(i)
        key_fixed = 'gcns.{}.mlp.0.weight_mask_fixed'.format(i)
        rewind_weight[key_train] = reverse_weight_mask(rewind_weight[key_train])
        rewind_weight[key_fixed] = reverse_weight_mask(rewind_weight[key_train])

    best_val_history = {'val_acc': 0, 'round': 0, 'epoch': 0, 'test_acc': 0}
    best_dict_history = {}
    rounds = 6
    origin_weight_temp = copy.deepcopy(rewind_weight)
    adv_weight_temp = copy.deepcopy(rewind_weight_mask_copy)

    for round in range(rounds):
        if round != 0:
            for i in range(num_layers):
                key_train = 'gcns.{}.mlp.0.weight_mask_train'.format(i)
                key_fixed = 'gcns.{}.mlp.0.weight_mask_fixed'.format(i)
                origin_weight_temp[key_train] = best_mask_refine_origin[key_train]
                origin_weight_temp[key_fixed] = best_mask_refine_origin[key_train]
                adv_weight_temp[key_train] = best_mask_refine_adv[key_train]
                adv_weight_temp[key_fixed] = best_mask_refine_adv[key_train]
        model_origin.load_state_dict(origin_weight_temp)
        model_adv.load_state_dict(adv_weight_temp)
        adj_spar, wei_spar = adv_pruning.get_sparsity(model_origin)
        recover_rate = 0.002 * wei_spar / 100
        optimizer_origin = optim.Adam(model_origin.parameters(), lr=args.lr)
        optimizer_adv = optim.Adam(model_adv.parameters(), lr=args.lr)
        best_val_acc_origin = {'highest_valid': 0, 'final_train': 0, 'final_test': 0, 'highest_train': 0, 'epoch': 0}
        best_val_acc_adv = {'highest_valid': 0, 'final_train': 0, 'final_test': 0, 'highest_train': 0, 'epoch': 0}
        for epoch in range(20):
            t0 = time.time()
            train_parts = dataset.random_partition_graph(dataset.total_no_of_nodes, cluster_number=args.cluster_number)
            data = dataset.generate_sub_graphs(train_parts, cluster_number=args.cluster_number, ifmask=True)
            epoch_loss_origin, adj_spar_origin, wei_spar_origin = train.train_mask(epoch, data, dataset, model_origin, optimizer_origin, criterion, device, args)
            epoch_loss_adv, adj_spar_adv, wei_spar_adv = train.train_mask(epoch, data, dataset, model_adv, optimizer_adv, criterion, device, args)

            result_origin = train.multi_evaluate(valid_data_list, dataset, model_origin, evaluator, device)
            result_adv = train.multi_evaluate(valid_data_list, dataset, model_adv, evaluator, device)


            train_result_origin = result_origin['train']['rocauc']
            valid_result_origin = result_origin['valid']['rocauc']
            test_result_origin = result_origin['test']['rocauc']
            train_result_adv = result_adv['train']['rocauc']
            valid_result_adv = result_adv['valid']['rocauc']
            test_result_adv = result_adv['test']['rocauc']

            if valid_result_adv > best_val_acc_adv['highest_valid']:
                best_val_acc_adv['highest_valid'] = valid_result_adv
                best_val_acc_adv['final_train'] = train_result_adv
                best_val_acc_adv['final_test'] = test_result_adv
                best_val_acc_adv['epoch'] = epoch
                wei_thre_index, best_epoch_mask_adv = adv_pruning.get_final_mask_epoch(model_adv, recover_rate)
                best_adv_model_copy = copy.deepcopy(model_adv)
                # pruning.save_all(model, rewind_weight_mask, optimizer, imp_num, epoch, args.model_save_path, 'IMP{}_train_ckpt'.format(imp_num))
            if valid_result_origin > best_val_acc_origin['highest_valid']:
                best_val_acc_origin['highest_valid'] = valid_result_origin
                best_val_acc_origin['final_train'] = train_result_origin
                best_val_acc_origin['final_test'] = test_result_origin
                best_val_acc_origin['epoch'] = epoch
                best_epoch_mask_origin = adv_pruning.get_final_mask_epoch_adv(model_origin, wei_thre_index)
                best_origin_model_copy = copy.deepcopy(model_origin)
        best_mask_refine_origin = adv_pruning.get_final_mask_round_origin(best_origin_model_copy,
                                                                          best_epoch_mask_adv,
                                                                          best_epoch_mask_origin)
        best_mask_refine_adv = adv_pruning.get_final_mask_round_adv(best_adv_model_copy,
                                                                    best_epoch_mask_adv,
                                                                    best_epoch_mask_origin)
        if best_val_acc_origin['highest_valid'] > best_val_history['highest_valid']:
            best_val_history['final_train'] = best_val_acc_origin['final_train']
            best_val_history['final_test'] = best_val_acc_origin['final_test']
            best_val_history['epoch'] = best_val_acc_origin['epoch']
            best_val_history['round'] = round
            best_dict_history = copy.deepcopy(best_mask_refine_origin)
    num_layers = args.num_layers
    for i in range(num_layers):
        key_train = 'gcns.{}.mlp.0.weight_mask_train'.format(i)
        key_fixed = 'gcns.{}.mlp.0.weight_mask_fixed'.format(i)
        rewind_weight[key_train] = best_dict_history[key_train]
        rewind_weight[key_fixed] = best_dict_history[key_fixed]
    model_origin.load_state_dict(rewind_weight)
    adj_spar, wei_spar = pruning.get_sparsity(model_origin)
    print(
        "Refine Mask : Best Val:[{:.2f}] at round:[{}] epoch:[{}] | Final Test Acc:[{:.2f}] Adj:[{:.2f}%] Wei:[{:.2f}%]".format(
            best_val_history['highest_valid'] * 100, best_val_history['round'], best_val_history['epoch'],
            best_val_history['highest_test'] * 100, adj_spar, wei_spar))
    best_acc_val, final_acc_test, final_epoch_list, adj_spar, wei_spar = main_fixed_mask(args, imp_num, rewind_weight)
    return best_acc_val, final_acc_test, final_epoch_list, adj_spar, wei_spar
    return final_state_dict

    
if __name__ == "__main__":

    args = ArgsInit().save_exp()
    imp_num = args.imp_num
    percent_list = [(1 - (1 - 0.05) ** (i + 1), 1 - (1 - 0.2) ** (i + 1)) for i in range(20)]
    
    args.pruning_percent_adj, args.pruning_percent_wei = percent_list[imp_num - 1]
    pruning.print_args(args, 80)
    pruning.setup_seed(666)
    print("INFO: IMP:[{}] Pruning adj[{:.6f}], wei[{:.6f}]".format(imp_num, args.pruning_percent_adj, args.pruning_percent_wei))
    
    resume_train_ckpt = None
    if args.resume_dir:
        resume_train_ckpt = torch.load(args.resume_dir)
        imp_num = resume_train_ckpt['imp_num']
        if 'fixed_ckpt' in args.resume_dir:
            main_fixed_mask(args, imp_num, final_state_dict=None, resume_train_ckpt=resume_train_ckpt)
            exit()

    # final_state_dict = main_get_mask(args, imp_num, resume_train_ckpt)
    # print("final_state_dict", final_state_dict)
    # print("INFO: Begin Retrain!")
    # main_fixed_mask(args, imp_num, final_state_dict=final_state_dict, resume_train_ckpt=None)
    with open('origin_ogb_arxiv_wei_wei.txt', "w") as f:
        for imp_num in range(1, 21):
            best_epoch_mask, rewind_weight_mask = main_get_mask(args, imp_num, rewind_weight_mask, resume_train_ckpt)
            num_layers = args.num_layers
            for i in range(num_layers):
                key_train = 'gcns.{}.mlp.0.weight_mask_train'.format(i)
                key_fixed = 'gcns.{}.mlp.0.weight_mask_fixed'.format(i)
                rewind_weight_mask[key_train] = best_epoch_mask[key_train]
                rewind_weight_mask[key_fixed] = best_epoch_mask[key_fixed]

            # # origin
            # valid, epoch, test_acc, adj_spar, wei_spar = main_fixed_mask(args, imp_num, rewind_weight_mask)
            # f.write("{:2f},{:2f},{:2f}\n".format((100 - adj_spar), (100 - wei_spar), test_acc * 100))

            # # adv
            rewind_weight_pass = copy.deepcopy(rewind_weight_mask)
            best_acc_val, final_acc_test, final_epoch_list, adj_spar, wei_spar = main_adv_train(args, rewind_weight_pass, imp_num)
            print("Adv-GLT : Sparsity:[{}], Best Val:[{:.2f}] at epoch:[{}] | Final Test Acc:[{:.2f}] Adj:[{:.2f}%] Wei:[{:.2f}%]"
                      .format(imp_num + 1, best_acc_val * 100, final_epoch_list, final_acc_test * 100, adj_spar, wei_spar))
            f.write("{:2f},{:2f},{:2f}\n".format((100 - adj_spar), (100 - wei_spar), final_acc_test * 100))


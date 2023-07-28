import os
import time

import dgl
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import net as net
from utils import load_data, load_adj_raw
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

def set_seed(seed):
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


def run_baseline_gcn(args, seed):
    set_seed(seed)
    adj, features, labels, idx_train, idx_val, idx_test = load_data(args['dataset'])
    adj = adj.cuda()
    features = features.cuda()
    labels = labels.cuda()
    loss_func = nn.CrossEntropyLoss()
    net_gcn = net.net_gcn_baseline(embedding_dim=args['embedding_dim'])
    net_gcn = net_gcn.cuda()
    optimizer = torch.optim.Adam(net_gcn.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    best_val_acc = {'val_acc': 0, 'epoch': 0, 'test_acc': 0}

    for epoch in range(args['epochs']):
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
            if acc_val > best_val_acc['val_acc']:
                best_val_acc['val_acc'] = acc_val
                best_val_acc['test_acc'] = acc_test
                best_val_acc['epoch'] = epoch

        print("Epoch:[{}] Val:[{:.2f}] Test:[{:.2f}] | Final Val:[{:.2f}] Test:[{:.2f}] at Epoch:[{}]"
              .format(epoch, acc_val * 100,
                      acc_test * 100,
                      best_val_acc['val_acc'] * 100,
                      best_val_acc['test_acc'] * 100,
                      best_val_acc['epoch']))
    torch.save(net_gcn.state_dict(), "Result/baseline.pkl")
    return best_val_acc['val_acc'], best_val_acc['test_acc'], best_val_acc['epoch']

def run_baseline_gin(args, seed):
    set_seed(seed)
    adj, features, labels, idx_train, idx_val, idx_test = load_data(args['dataset'])
    adj = load_adj_raw(args['dataset'])
    node_num = features.size()[0]
    g = dgl.DGLGraph()
    g.add_nodes(node_num)
    adj = adj.tocoo()
    g.add_edges(adj.row, adj.col)
    features = features.cuda()
    labels = labels.cuda()
    g = g.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    loss_func = nn.CrossEntropyLoss()
    net_gin = net.net_gin_baseline(args['embedding_dim'], g)
    net_gin = net_gin.cuda()
    optimizer = torch.optim.Adam(net_gin.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    best_val_acc = {'val_acc': 0, 'epoch': 0, 'test_acc': 0}
    for epoch in range(args['epochs']):

        optimizer.zero_grad()
        output = net_gin(g, features, 0, 0)
        loss = loss_func(output[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            net_gin.eval()
            output = net_gin(g, features, 0, 0)
            acc_val = f1_score(labels[idx_val].cpu().numpy(), output[idx_val].cpu().numpy().argmax(axis=1),
                               average='micro')
            acc_test = f1_score(labels[idx_test].cpu().numpy(), output[idx_test].cpu().numpy().argmax(axis=1),
                                average='micro')
            if acc_val > best_val_acc['val_acc']:
                best_val_acc['val_acc'] = acc_val
                best_val_acc['test_acc'] = acc_test
                best_val_acc['epoch'] = epoch
        print("Epoch:[{}] LOSS:[{:.2f}] Val:[{:.2f}] Test:[{:.2f}] | Final Val:[{:.2f}] Test:[{:.2f}] at Epoch:[{}]".format(epoch, loss,
               acc_val * 100,
               acc_test * 100,
               best_val_acc['val_acc'] * 100,
               best_val_acc['test_acc'] * 100,
               best_val_acc['epoch']))
    return best_val_acc['val_acc'], best_val_acc['test_acc'], best_val_acc['epoch']

def run_baseline_gat(args, seed):
    set_seed(seed)
    adj, features, labels, idx_train, idx_val, idx_test = load_data(args['dataset'])
    adj = load_adj_raw(args['dataset'])

    node_num = features.size()[0]

    g = dgl.DGLGraph()
    g.add_nodes(node_num)
    adj = adj.tocoo()
    g.add_edges(adj.row, adj.col)
    features = features.cuda()
    labels = labels.cuda()
    g = g.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    loss_func = nn.CrossEntropyLoss()
    net_gat = net.net_gat_baseline(args['embedding_dim'], g)
    g.add_edges(list(range(node_num)), list(range(node_num)))
    net_gat = net_gat.cuda()
    optimizer = torch.optim.Adam(net_gat.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    best_val_acc = {'val_acc': 0, 'epoch': 0, 'test_acc': 0}
    for epoch in range(args['epochs']):
        optimizer.zero_grad()
        output = net_gat(g, features, 0, 0)
        loss = loss_func(output[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            net_gat.eval()
            output = net_gat(g, features, 0, 0)
            acc_val = f1_score(labels[idx_val].cpu().numpy(), output[idx_val].cpu().numpy().argmax(axis=1),
                               average='micro')
            acc_test = f1_score(labels[idx_test].cpu().numpy(), output[idx_test].cpu().numpy().argmax(axis=1),
                                average='micro')
            if acc_val > best_val_acc['val_acc']:
                best_val_acc['val_acc'] = acc_val
                best_val_acc['test_acc'] = acc_test
                best_val_acc['epoch'] = epoch

        print("Epoch:[{}] LOSS:[{:.2f}] Val:[{:.2f}] Test:[{:.2f}] | Final Val:[{:.2f}] Test:[{:.2f}] at Epoch:[{}]"
               .format(epoch, loss,
                acc_val * 100,
                acc_test * 100,
                best_val_acc['val_acc'] * 100,
                best_val_acc['test_acc'] * 100,
                best_val_acc['epoch']))
    return best_val_acc['val_acc'], best_val_acc['test_acc'], best_val_acc['epoch']



def parser_loader():
    parser = argparse.ArgumentParser(description='baseline')
    """Other settings"""
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--embedding-dim', nargs='+', type=int, default=[1433, 512, 7])
    parser.add_argument('--lr', type=float, default=0.008)
    parser.add_argument('--weight-decay', type=float, default=8e-5)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--backbone', type=str, default='gcn')
    return parser

if __name__ == "__main__":
    parser = parser_loader()
    args = vars(parser.parse_args())
    print(args)
    seed_dict = {'cora': 2377, 'citeseer': 4428, 'pubmed': 3333}
    seed = seed_dict[args['dataset']]
    start = time.clock()
    if args['backbone'] == 'gcn':
        best_acc_val, final_acc_test, final_epoch_list = run_baseline_gcn(args, seed)
    elif args['backbone'] == 'gin':
        best_acc_val, final_acc_test, final_epoch_list = run_baseline_gin(args, seed)
    else:
        best_acc_val, final_acc_test, final_epoch_list = run_baseline_gat(args, seed)
    print("=" * 120)
    print("syd : Best Val:[{:.2f}] at epoch:[{}] | Final Test Acc:[{:.2f}]"
          .format(best_acc_val * 100, final_epoch_list, final_acc_test * 100, ))
    print("=" * 120)
    end = time.clock()
    print("run time: ", end-start)
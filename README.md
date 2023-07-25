# [ECAI 2023] Adversarial Erasing with Pruned Elements: Towards Better Graph Lottery Tickets
[![License: Apache](https://img.shields.io/badge/License-Apache-blue.svg)](LICENSE)

Official codebase for paper [Adversarial Erasing with Pruned Elements: Towards Better Graph Lottery Tickets](https://arxiv.org/abs/xxx.xxxx). This codebase is based on the open-source [DGL](https://docs.dgl.ai/) framework and please refer to that repo for more documentation.

## Overview
**Abstract:**
Graph Lottery Ticket (GLT), a combination of core subgraph and sparse subnetwork, has been proposed to mitigate the computational cost of deep Graph Neural Networks (GNNs) on large input graphs while preserving original performance.
However, the winning GLTs in existing studies are obtained by applying iterative magnitude-based pruning (IMP) without re-evaluating and re-considering the pruned information, which disregards the dynamic changes in the significance of edges/weights during graph/model structure pruning, and thus limits the appeal of the winning tickets. 
In this paper, we formulate a conjecture, i.e., existing overlooked valuable information in the pruned graph connections and model parameters which can be re-grouped into GLT to enhance the final performance.
Specifically, we propose an *adversarial complementary erasing* (ACE) framework to explore the valuable information from the pruned components, thereby developing a more powerful GLT, referred to as the **ACE-GLT**. The main idea is to mine valuable information from pruned edges/weights after each round of IMP, and employ the ACE technique to refine the GLT processing. Finally, experimental results demonstrate that our ACE-GLT outperforms existing methods for searching GLT in diverse tasks.
<div align="center">
<img src="https://github.com/Wangyuwen0627/ACE-GLT/blob/master/Figs/method.png" width="100%">
</div>

## Prerequisites
### Install dependencies
See `requirment.txt` file for more information about how to install the dependencies.

## Usage
Please follow the instructions below to replicate the results in the paper.
```
# baseline(non-pruned cases)
python baseline.py --dataset <dataset> --embedding-dim <feature_dim [hidden_dims] output_dims> --backbone <backbone>

# NodeClassification
python glt_gcn.py --dataset <dataset> --embedding-dim <feature_dim [hidden_dims] output_dims> --lr 0.008 --weight-decay 8e-5 --pruning_percent_wei 0.2 --pruning_percent_adj 0.05 --mask_epochs 200 --fix_epochs 200 --s1 1e-2 --s2 1e-2 --init_soft_mask_type all_one
python glt_gin.py --dataset <dataset> --embedding-dim <feature_dim [hidden_dims] output_dims> --lr 0.008 --weight-decay 8e-5 --pruning_percent_wei 0.2 --pruning_percent_adj 0.05 --mask_epochs 200 --fix_epochs 200 --s1 1e-3 --s2 1e-3
python glt_gat.py --dataset <dataset> --embedding-dim <feature_dim [hidden_dims] output_dims> --lr 0.008 --weight-decay 8e-5 --pruning_percent_wei 0.2 --pruning_percent_adj 0.05 --mask_epochs 200 --fix_epochs 200 --s1 1e-3 --s2 1e-3

```

## Citation



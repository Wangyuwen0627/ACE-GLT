# [ECAI 2023] Adversarial Erasing with Pruned Elements: Towards Better Graph Lottery Tickets
[![License: Apache](https://img.shields.io/badge/License-Apache-blue.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2308.02916-b31b1b.svg)](https://arxiv.org/abs/2308.02916)

Official codebase for paper [Adversarial Erasing with Pruned Elements: Towards Better Graph Lottery Tickets](https://arxiv.org/abs/xxx.xxxx). This codebase is based on the open-source [DGL](https://docs.dgl.ai/) framework and please refer to that repo for more documentation.

## Overview
**Abstract:**
Graph Lottery Ticket (GLT), a combination of core subgraph and sparse subnetwork, has been proposed to mitigate the computational cost of deep Graph Neural Networks (GNNs) on large input graphs while preserving original performance.
However, the winning GLTs in existing studies are obtained by applying iterative magnitude-based pruning (IMP) without re-evaluating and re-considering the pruned information, which disregards the dynamic changes in the significance of edges/weights during graph/model structure pruning, and thus limits the appeal of the winning tickets. 
In this paper, we formulate a conjecture, i.e., existing overlooked valuable information in the pruned graph connections and model parameters which can be re-grouped into GLT to enhance the final performance.
Specifically, we propose an *adversarial complementary erasing* (ACE) framework to explore the valuable information from the pruned components, thereby developing a more powerful GLT, referred to as the **ACE-GLT**. The main idea is to mine valuable information from pruned edges/weights after each round of IMP, and employ the ACE technique to refine the GLT processing. Finally, experimental results demonstrate that our ACE-GLT outperforms existing methods for searching GLT in diverse tasks.
<div align="center">
<img src="https://github.com/Wangyuwen0627/ACE-GLT/blob/main/Figs/method.png" width="100%">
</div>

## Prerequisites
### Install dependencies
See `requirement.txt` file for more information about how to install the dependencies.

## Usage
Please follow the instructions below to replicate the results in the paper.
### Small-scale datasets
```
# baseline (non-pruned cases)
python small_scale/baseline.py --dataset <dataset> --embedding-dim <feature_dim [hidden_dims] output_dims> --backbone <backbone>

# ACE-GLT
python small_scale/glt_gcn.py --dataset <dataset> --pruning_percent_wei 0.2 --pruning_percent_adj 0.05 --mask_epochs 200 --fix_epochs 200 --s1 1e-2 --s2 1e-2 --init_soft_mask_type all_one
python small_scale/glt_gin.py --dataset <dataset> --pruning_percent_wei 0.2 --pruning_percent_adj 0.05 --mask_epochs 200 --fix_epochs 200 --s1 1e-3 --s2 1e-3 --init_soft_mask_type all_one
python small_scale/glt_gat.py --dataset <dataset> --pruning_percent_wei 0.2 --pruning_percent_adj 0.05 --mask_epochs 200 --fix_epochs 200 --s1 1e-3 --s2 1e-3 --init_soft_mask_type all_one
```

<div align="center">
<img src="https://github.com/Wangyuwen0627/ACE-GLT/blob/main/Figs/small_scale.png" width="100%">
</div>

### Large-scale datasets
```
# baseline (non-pruned cases)
python large_scale/<dataset>/baseline.py

# ACE-GLT
python large_scale/ogbn-arxiv/glt_resgcn.py --use_gpu --self_loop --learn_t --num_layers 28 --block res+ --gcn_aggr softmax_sg --t 0.1 --s1 1e-6 --s2 1e-4 --pruning_percent_wei 0.2 --pruning_percent_adj 0.05 --mask_epochs 250 --fix_epochs 500 --model_save_path IMP
python large_scale/ogbn-proteins/glt_resgcn.py --use_gpu --conv_encode_edge --use_one_hot_encoding --learn_t --num_layers 28 --s1 1e-1 --s2 1e-3 --pruning_percent_wei 0.2 --pruning_percent_adj 0.05 --mask_epochs 250 --fix_epochs 500 --model_save_path IMP
```

<div align="center">
<img src="https://github.com/Wangyuwen0627/ACE-GLT/blob/main/Figs/large_scale.png" width="100%">
</div>

## Citation
If you find this work useful for your research, please cite our paper:

```
@article{wang2023adversarial,
      title={Adversarial Erasing with Pruned Elements: Towards Better Graph Lottery Ticket}, 
      author={Yuwen Wang and Shunyu Liu and Kaixuan Chen and Tongtian Zhu and Ji Qiao and Mengjie Shi and Yuanyu Wan and Mingli Song},
      journal={arXiv preprint arXiv:2308.02916},
      year={2023},
}
```

## Contact
Please feel free to contact me via email (<yuwenwang@zju.edu.cn>) if you have any questions about our work.



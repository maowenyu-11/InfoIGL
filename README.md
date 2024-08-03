>📋  A  README.md for code accompanying our paper InfoIGL

# Invariant Graph Learning Meets Information Bottleneck for Out-of-Distribution Generalization



## Requirements

Main packages: PyTorch, Pytorch Geometric, OGB. To prepare the environment:

```
pytorch==1.10.1
torch-cluster==1.5.9
torch-geometric==2.0.3
torch-scatter==2.0.9
torch-sparse==0.6.12
torch-spline-conv==1.2.1
ogb==1.3.4
```

>📋 Please download the [graph OOD datasets](https://github.com/divelab/GOOD/) and [OGB datasets](https://ogb.stanford.edu/) as described in the original paper. Create a folder dataset, and then put the datasets into dataset. Then modify the path by specifying --root /dataset

## Training

 We use the NVIDIA GeForce RTX 3090 (24GB GPU) to conduct all our experiments. To run the code on Motif, please use the following command:

```
python -u  main_mol.py \
 --dataset motif \
 --domain basis \
 --shift concept \
 --lr 1e-3 \
 --min_lr 1e-6 \
 --weight_decay 0.1 \
 --hidden 128 \
 --epoch 200 \
 --batch_size 128 \
 --trails 10 \
 --pretrain 40 \
 --constraint 0.7 \
 --layers 3 \
 --g 0.5 \
 --l 0.5 \

```
```
 python -u  main_mol.py \
 --dataset motif \
 --domain size \
 --shift covariate \
 --lr 1e-3 \
 --min_lr 1e-3 \
 --weight_decay 0 \
 --hidden 64 \
 --epoch 200 \
 --batch_size 1024 \
 --trails 10 \
 --pretrain 40 \
 --constraint 0.7 \
 --layers 3 \
 --g 1 \
 --l 0.2 >>
```
>📋  To run the code on HIV, please use the following command:
```
python -u  main_hiv_size.py \
 --dataset hiv \
 --domain size \
 --lr=0.001 \
 --min_lr 1e-6 \
 --weight_decay 0.01 \
 --hidden 300 \
 --epoch 100 \
 --batch_size 256 \
 --trails 10 \
 --pretrain 80 \
 --constraint 0.7 \
 --g 0.5 \
 --l 0.5
```
```
python -u  main_hiv_scaffold.py \
 --dataset hiv \
 --domain scaffold \
 --shift concept \
 --lr 1e-2 \
 --min_lr 1e-6 \
 --weight_decay 1e-5 \
 --hidden 128 \
 --epoch 200 \
 --batch_size 1024 \
 --trails 10 \
 --pretrain 40 \
 --layers 3 \
 --pred 1 \
 --g 1 \
 --l 0.1
```
>📋  To run the code on Molbbbp, please use the following command:
```
python -u  main_bbbp.py \
 --dataset molbbbp \
 --domain size \
 --lr 1e-2 \
 --min_lr 1e-6 \
 --weight_decay 1e-5 \
 --hidden 128 \
 --epoch 100 \
 --batch_size 64 \
 --trails 10 \
 --pretrain 20 \
 --constraint 0.2 \
 --layers 2 \
 --g 0.5 \
 --l 0.2
```
```
python -u main_bbbp.py \
--dataset ogbg-molbbbp \
--domain scafold \
--batch_size 1024 \
--lr 0.0001 \
--hidden 300 \
--weight_decay 0 \
--trails 10 \
--pretrain 80 \
--constraint 0.7 \
--epochs 100 \
--g 0.2 \
--l 0.2 

```
>📋  To run the code on CMNIST, please use the following command:
```
python -u  main_mol.py \
 --dataset cmnist \
 --domain color \
 --shift covariate \
 --lr 1e-3 \
 --min_lr 1e-3 \
 --weight_decay 0 \
 --hidden 32 \
 --epoch 150 \
 --batch_size 256 \
 --trails 10 \
 --pretrain 60 \
 --layers 5 \
 --constraint 0.7 \
 --g 0.5 \
 --l 0.1 

```

 

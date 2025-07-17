#!/usr/bin/env bash

function train_mnist {
    local vit=$1
    local te=$2
    local precision=$3
    local epoch=$4
    echo "USE_TORCHVISION_VIT=$vit, USE_TE_LINEAR=$te, precision=$precision, EPOCH=$epoch"
    USE_TORCHVISION_VIT=$vit USE_TE_LINEAR=$te NEPOCH=$epoch python ${precision}_train_mnist.py
}

n_epoch=10

# old signature
# train_mnist 1 $n_epoch fp32
# train_mnist 1 $n_epoch fp16
# train_mnist 0 $n_epoch fp32
# train_mnist 0 $n_epoch fp16

# Sample results:
# | USE\_TORCHVISION\_VIT | Epoch | Precision | Train Loss | Train Acc (%) | Test Acc (%) |
# | :-------------------: | :---: | :-------: | :--------: | :-----------: | :----------: |
# |           1           |   10  |    fp32   |   0.2432   |     92.28     |     95.03    |
# |           0           |   10  |    fp32   |   0.2010   |     93.65     |     95.88    |

# |           1           |   10  |    fp16   |   0.2345   |     92.60     |     95.13    |
# |           0           |   10  |    fp16   |   0.1912   |     93.87     |     96.40    |

# this script is created to test if both torchvision and custom ViT models work correctly with fp32 and fp16 training. 
# Yes per results above. interesingly, torchvision ViT performs slightly worse than custom ViT in both fp32 and fp16 training. I think this is just variance, not significant enough.


# torchvision ViT
train_mnist 1 0 fp32 $n_epoch
train_mnist 1 0 fp16 $n_epoch
# custom ViT
train_mnist 0 0 fp32 $n_epoch
train_mnist 0 0 fp16 $n_epoch
# custom ViT with te.Linear
train_mnist 0 1 fp32 $n_epoch
train_mnist 0 1 fp16 $n_epoch

# | USE\_TORCHVISION\_VIT | USE\_TE\_LINEAR | Precision | Train Loss | Train Acc (%) | Test Acc (%) |
# | :-------------------: | :-------------: | :-------: | :--------: | :-----------: | :----------: |
# |           1           |        0        |    fp32   |   0.2344   |     92.40     |     94.98    |
# |           0           |        0        |    fp32   |   0.2002   |     93.55     |     95.27    |
# |           0           |        1        |    fp32   |   0.2112   |     93.08     |     95.92    |

# |           1           |        0        |    fp16   |   0.2386   |     92.39     |     94.99    |
# |           0           |        0        |    fp16   |   0.1734   |     94.50     |     96.71    |
# |           0           |        1        |    fp16   |   0.2003   |     93.50     |     95.42    |

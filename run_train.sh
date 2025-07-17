#!/usr/bin/env bash

function train_mnist {
    local vit=$1
    local epoch=$2
    local precision=$3
    echo "Training ... USE_TORCHVISION_VIT=$vit, EPOCH=$epoch, precision=$precision"
    USE_TORCHVISION_VIT=$vit NEPOCH=$epoch python ${precision}_train_mnist.py
}

n_epoch=10

train_mnist 1 $n_epoch fp32
train_mnist 1 $n_epoch fp16
train_mnist 0 $n_epoch fp32
train_mnist 0 $n_epoch fp16

# Sample results:
# | USE\_TORCHVISION\_VIT | Epoch | Precision | Train Loss | Train Acc (%) | Test Acc (%) |
# | :-------------------: | :---: | :-------: | :--------: | :-----------: | :----------: |
# |           1           |   10  |    fp32   |   0.2432   |     92.28     |     95.03    |
# |           0           |   10  |    fp32   |   0.2010   |     93.65     |     95.88    |

# |           1           |   10  |    fp16   |   0.2345   |     92.60     |     95.13    |
# |           0           |   10  |    fp16   |   0.1912   |     93.87     |     96.40    |

# this script is created to test if both torchvision and custom ViT models work correctly with fp32 and fp16 training. 
# Yes per results above. interesingly, torchvision ViT performs slightly worse than custom ViT in both fp32 and fp16 training. I think this is just variance, not significant enough.
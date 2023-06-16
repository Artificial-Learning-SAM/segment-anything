import os
import argparse
import sys
sys.path.append("..")

import numpy as np
import nibabel as nib
import cv2
import matplotlib as mpl
import torch

from segment_anything import sam_model_registry, SamPredictor
from data_utils import DataLoader, GetPointsFromMask, GetBBoxFromMask
from segment_anything.modeling.mask_decoder import MLP

parser = argparse.ArgumentParser(description='Task 3')
parser.add_argument('-p', '--prompt', type=str,
                    help='List of numbers. x>0 means sampling x points from mask. x<0 means sampling x points, but using center (max distance to boundary) of mask as the first point. x==0 means using bbox. E.g. "[0, 1, -1, 3]".')
parser.add_argument('-e', '--epoch', type=int,
                    help='Number of training epoch',
                    default=300)
parser.add_argument('-bs', '--batch_size', type=int,
                    help='Batch size',
                    default=64)
parser.add_argument('--device', type=str,
                    help='Device to use (cpu or cuda)',
                    default='cuda')
args = parser.parse_args()
args.prompt = eval(args.prompt)

print("Imports done")

# Init SAM

model_type = "vit_h"
import inspect
sam_checkpoint = "../sam_vit_h_4b8939.pth"
sam = sam_model_registry[model_type](checkpoint=None , decoder_version="task1")
with open(sam_checkpoint, 'rb') as f:
    pre_dict = torch.load(f, map_location='cuda')
class_dict = sam.state_dict()
for name, param in pre_dict.items():
    if name in class_dict:
        class_dict[name].copy_(param)
sam.load_state_dict(class_dict)
sam_checkpoint = "./epoch-119-val-0.8411638965.pth"

try:
    with open(sam_checkpoint, 'rb') as f:
        state_dict = torch.load(f, map_location='cuda')
        sam.mask_decoder.load_state_dict(state_dict)
except:
    print('cannot find',sam_checkpoint , 'as decoder weight')
sam.to(device=args.device)
sam.train()
print("SAM initialized")
# Start training
from statistics import mean
from torch.nn.functional import threshold, normalize
from tqdm import tqdm
import torchvision.transforms as tfs
from utils import *

from segment_anything.modeling.mask_decoder_3 import MaskDecoder3
from segment_anything.modeling.transformer import TwoWayTransformer
prompt_embed_dim = 256
class_decoder=MaskDecoder3(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )
class_decoder.to(device=args.device)
print('Start training')

lr = 1e-5
# wd = 1e-5
# 使用自适应的学习率
# optimizer = torch.optim.AdamW(sam.mask_decoder.parameters(), lr=lr, weight_decay=wd)
optimizer = torch.optim.Adam(sam.mask_decoder.parameters(), lr=lr)
#loss_fn = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
#use MSE loss
loss_fn = torch.nn.MSELoss(reduction='mean')
#output type of sam.mask_decoder
# data augmentation
my_transform = tfs.Compose([
            tfs.RandomHorizontalFlip(p=0.5), 
            tfs.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3)
        ])

np.random.seed(0)
dataloader = DataLoader('train', sam, args)
dataloader_val = DataLoader('val', sam, args)

losses = []
accurate = []
accurate_val = []

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(type(sam.mask_decoder))

# Do one epoch, mode can be 'train' or 'val'
def do_epoch(epoch, dataloader, mode):
    epoch_loss = []
    epoch_accurate_rate = []

    for i in tqdm(range(len(dataloader))):
        image_embeddings, sparse_embeddings, dense_embeddings, gt_masks, organ = dataloader.get_batch()
        class_logits = class_decoder(
            image_embeddings=image_embeddings,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        gt_organ_class = torch.zeros((len(organ), 14))
        # print(organ)
        for j in range(len(organ)):
            # assert (1 <= organ[j] <= 14)
            gt_organ_class[j][organ[j]] = 1

        loss = loss_fn(class_logits, gt_organ_class.to(device=args.device))
        epoch_loss.append(loss.item())
        if mode == 'train':
            # loss = loss_fn(classifier_predictions, gt_organ_class.to(device=args.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        accurate = torch.argmax(class_logits, dim=1) == torch.argmax(gt_organ_class.to(device=args.device), dim=1)
        accurate_rate = torch.sum(accurate).item() / len(accurate)
        epoch_accurate_rate.append(accurate_rate)
        # print(f'loss: {loss.item()}')
        # print(f'accurate rate: {accurate_rate}')

    if mode == 'train':
        print(f'Epoch:{epoch}')
        print(f'loss: {mean(epoch_loss)}')
        print(f'accurate rate: {mean(epoch_accurate_rate)}')

    if mode == 'train':
        return epoch_loss, epoch_accurate_rate
    else:
        print(f'Val Epoch:{epoch}')
        print(f'loss: {mean(epoch_loss)}')
        print(f'accurate rate: {mean(epoch_accurate_rate)}')
        return epoch_loss , epoch_accurate_rate


# Training
for epoch in range(args.epoch):
    epoch_loss , epoch_accurate  = do_epoch(epoch, dataloader, 'train')
    losses.append(mean(epoch_loss))
    accurate.append(mean(epoch_accurate))

    
    # Validation
    with torch.no_grad():
        epoch_loss, epoch_accurate_rate = do_epoch(epoch, dataloader_val, 'val')
    accurate_val.append(mean(epoch_accurate_rate))
    torch.save(class_decoder.state_dict(), f'./model/class_decoder-epoch-{epoch}-val-{mean(epoch_accurate_rate):.5f}.pth')
    # Save model
    print(losses)
    print(accurate)
    print(accurate_val)
    #restore to log.txt
    with open('log.txt', 'a') as f:
        f.write(f'Epoch:{epoch}\n')
        f.write(f'loss: {mean(epoch_loss)}\n')
        f.write(f'accurate rate: {mean(epoch_accurate_rate)}\n')
    # Plot loss and dice
    plot_curve(losses, accurate, accurate_val)



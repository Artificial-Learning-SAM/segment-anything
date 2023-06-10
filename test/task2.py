import os
import argparse
import sys
sys.path.append("..")

import numpy as np
import nibabel as nib
import cv2
import matplotlib as mpl
import torch
from monai.losses import DiceCELoss

from segment_anything import sam_model_registry, SamPredictor
from data_utils import DataLoader, GetPointsFromMask, GetBBoxFromMask

parser = argparse.ArgumentParser(description='Task 1')
parser.add_argument('-n', '--number', type=int,
                    help='Number of points to sample from mask')
parser.add_argument('-c', '--center',
                    help='Use center (max distance to boundary) of mask as the first prompt',
                    action='store_true')
parser.add_argument('-b', '--bbox',
                    help='Use bounding box of mask as prompt',
                    action='store_true')
parser.add_argument('-e', '--epoch', type=int,
                    help='Number of training epoch',
                    default=100)
parser.add_argument('-bs', '--batch_size', type=int,
                    help='Batch size',
                    default=10)
parser.add_argument('--device', type=str,
                    help='Device to use (cpu or cuda)',
                    default='cuda')
args = parser.parse_args()

print("Imports done")

# Init SAM
sam_checkpoint = "../sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint, decoder_version="task2")
sam.to(device=args.device)
sam.train()
print("SAM initialized")

# Start training
from statistics import mean
from torch.nn.functional import threshold, normalize
from tqdm import tqdm
import torchvision.transforms as tfs
from utils import *

print('Start training')

lr = 1e-5
wd = 0
# 使用自适应的学习率
# optimizer = torch.optim.AdamW(sam.mask_decoder.parameters(), lr=lr, weight_decay=wd)
optimizer = torch.optim.Adam(sam.mask_decoder.parameters(), lr=lr)
loss_fn = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
# data augmentation
my_transform = tfs.Compose([
            tfs.RandomHorizontalFlip(p=0.5), 
            tfs.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3)
        ])

np.random.seed(0)
dataloader = DataLoader('train', sam, args)
dataloader_val = DataLoader('val', sam, args)

losses = []
dices = []
dices_val = []

# Do one epoch, mode can be 'train' or 'val'
def do_epoch(epoch, dataloader, mode):
    epoch_loss = []
    epoch_dice = {}
    for k in range(1, 14):
        epoch_dice[k] = []

    for i in tqdm(range(len(dataloader))):
        image_embeddings, sparse_embeddings, dense_embeddings, gt_masks, organ = dataloader.get_batch()
        low_res_masks, iou_predictions = sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        input_size, original_size = dataloader.input_size, dataloader.original_size
        upscaled_masks = sam.postprocess_masks(low_res_masks, input_size, original_size).to(args.device)
        gt_binary_mask = torch.as_tensor(gt_masks > 0, dtype=torch.float32)[:, None, :, :]

        if mode == 'train':
            loss = loss_fn(upscaled_masks, gt_binary_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

        for j in range(len(organ)):
            pred = upscaled_masks[j] > 0
            gt = gt_binary_mask[j]
            dice = 2 * torch.sum(pred * gt) / (torch.sum(pred) + torch.sum(gt))
            epoch_dice[organ[j]].append(dice.item())

    if mode == 'train':
        print(f'Epoch:{epoch}')
        print(f'loss: {mean(epoch_loss)}')

    for k in range(1, 14):
        if len(epoch_dice[k]) != 0:
            epoch_dice[k] = mean(epoch_dice[k])
        else:
            epoch_dice[k] = 0

    if mode == 'train':
        print(f'dice: {epoch_dice}')
        print(f'mean dice: {mean(epoch_dice.values())}')
        return epoch_loss, mean(epoch_dice.values())
    else:
        print(f'val dice: {epoch_dice}')
        print(f'val mean dice: {mean(epoch_dice.values())}')
        return mean(epoch_dice.values())


# Training
for epoch in range(args.epoch):
    epoch_loss, epoch_dice = do_epoch(epoch, dataloader, 'train')
    losses.append(mean(epoch_loss))
    dices.append(epoch_dice)

    # Validation
    epoch_dice = do_epoch(epoch, dataloader_val, 'val')
    dices_val.append(epoch_dice)

    # Save model
    torch.save(sam.mask_decoder.state_dict(), f'./model/epoch-{epoch}-val-{epoch_dice:.10f}.pth')

    # Plot loss and dice
    plot_curve(losses, dices, dices_val)

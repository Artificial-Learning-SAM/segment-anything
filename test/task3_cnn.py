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
from data_utils_cnn import DataLoader, GetPointsFromMask, GetBBoxFromMask

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
                    default=300)
parser.add_argument('-bs', '--batch_size', type=int,
                    help='Batch size',
                    default=64)
parser.add_argument('--device', type=str,
                    help='Device to use (cpu or cuda)',
                    default='cuda')
parser.add_argument('--decoder_weight', type=str,
                    help='Path to decoder weight',
                    default=None)
args = parser.parse_args()

print("Imports done")

# Init SAM
sam_checkpoint = "../sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint, decoder_version="task2")
sam.to(device=args.device)
if args.decoder_weight is not None:
    sam.mask_decoder.load_state_dict(torch.load(args.decoder_weight))
print("SAM initialized")

# Start training
from statistics import mean
from torch.nn.functional import threshold, normalize
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import torchvision.transforms as tfs
from utils import *

print('Start training')

from cnn_network import vgg11_bn

# Init vgg
vgg = vgg11_bn()
vgg = vgg.to(device=args.device)

lr = 1e-5
wd = 0
# 使用自适应的学习率
optimizer = torch.optim.Adam(vgg.parameters(), lr=lr)
loss_fn = CrossEntropyLoss()
# data augmentation
my_transform = tfs.Compose([
            tfs.RandomHorizontalFlip(p=0.5), 
            tfs.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3)
        ])

np.random.seed(0)
dataloader = DataLoader('train', sam, args)
dataloader_val = DataLoader('val', sam, args)

losses = []
acc = []
acc_val = []


# Do one epoch, mode can be 'train' or 'val'
def do_epoch(epoch, dataloader, mode):
    epoch_loss = []
    epoch_dice = {}
    acc = 0
    for k in range(1, 14):
        epoch_dice[k] = []

    for i in tqdm(range(len(dataloader))):
        image_embeddings, sparse_embeddings, dense_embeddings, gt_masks, organ, img_cnn= dataloader.get_batch(get_img = True)
        # print(image_embeddings.size())
        # torch.Size([2, 256, 64, 64])
        # bs*256*64*64
        #torch.Size([1, 256, 64, 64])
        with torch.no_grad():
            low_res_masks, iou_predictions = sam.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            # print(f'low_res:{low_res_masks.size()}')
            # low_res:torch.Size([2, 1, 256, 256])

            input_size, original_size = dataloader.input_size, dataloader.original_size
            upscaled_masks = sam.postprocess_masks(low_res_masks, input_size, original_size).to(args.device)
            binary_masks = upscaled_masks > 0
            # print(f'up:{upscaled_masks.size()}')
            # ([2, 1, 512, 512])
            gt_binary_masks = torch.as_tensor(gt_masks > 0, dtype=torch.float32)[:, None, :, :]

        # img_cnn: bs*1*512*512
        # binary_mask: bs*1*512*512
        # bs*2*512*512

        if mode == 'train':
            vgg.train()
            input = torch.cat((img_cnn[:, None, :, :], gt_binary_masks), dim=1)
            # input = gt_binary_masks
            input = input.float()
            output = vgg(input)

            label_organ = torch.as_tensor(organ, device=args.device) # crossentropyloss的target是从0开始的
            label_organ = label_organ - 1
            loss = loss_fn(output, label_organ)
            # print(f'loss:{loss.item()}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
            acc += torch.sum(torch.argmax(output, dim=1) == label_organ).item() / args.batch_size
        
        elif mode == 'val':
            vgg.eval()
            input = torch.cat((img_cnn[:, None, :, :], binary_masks), dim=1)
            # input = gt_binary_masks
            input = input.float()
            output = vgg(input)

            label_organ = torch.as_tensor(organ, device=args.device) # crossentropyloss的target是从0开始的
            label_organ = label_organ - 1
            loss = loss_fn(output, label_organ)
            
            epoch_loss.append(loss.item())
            acc += torch.sum(torch.argmax(output, dim=1) == label_organ).item() / args.batch_size


    if mode == 'train':
        print(f'Epoch:{epoch}')
        print(f'loss: {mean(epoch_loss)}')
        print(f'acc: {acc / len(dataloader)}')
        return epoch_loss, acc / len(dataloader)


    elif mode == 'val':
        print(f'Epoch:{epoch}')
        print(f'val loss: {mean(epoch_loss)}')
        print(f'val acc: {acc / len(dataloader)}')
        return epoch_loss, acc / len(dataloader)
        


# Training
for epoch in range(args.epoch):
    epoch_loss, epoch_acc = do_epoch(epoch, dataloader, 'train')
    losses.append(mean(epoch_loss))
    acc.append(epoch_acc)

    # Validation
    with torch.no_grad():
        epoch_val_loss, epoch_val_acc = do_epoch(epoch, dataloader_val, 'val')
    acc_val.append(epoch_val_acc)

    # # Save model
    # torch.save(sam.mask_decoder.state_dict(), f'./model/epoch-{epoch}-val-{epoch_dice:.10f}.pth')

    # Plot loss and dice
    plot_curve(losses, acc, acc_val, 'Acc')

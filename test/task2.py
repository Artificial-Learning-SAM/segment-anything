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
args = parser.parse_args()

print("Imports done")

# Init SAM
sam_checkpoint = "../sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
sam.train()
print("SAM initialized")

# Start training
from statistics import mean
from torch.nn.functional import threshold, normalize
from tqdm import tqdm
import torchvision.transforms as tfs
from segment_anything.utils.transforms import ResizeLongestSide
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
dataloader = DataLoader('train')
dataloader_val = DataLoader('val')

losses = []
dices = []
dices_val = []

# Do one epoch, mode can be 'train' or 'val'
def do_epoch(epoch, dataloader, mode):
    epoch_loss = []
    epoch_dice = {}
    for k in range(1, 14):
        epoch_dice[k] = []

    for i in tqdm(range(dataloader.slice_num())):
        image, label = dataloader.get_slice()

        transform = ResizeLongestSide(sam.image_encoder.img_size)
        transformed_image = transform.apply_image(image)
        transformed_image = torch.as_tensor(transformed_image, device=device).permute(2, 0, 1).contiguous()
        
        # Data augmentation
        # input_image_torch = my_transform(input_image_torch)

        transformed_image = transformed_image[None, :, :, :]#将图像转换成模型需要的格式
        input_image = sam.preprocess(transformed_image)

        input_size = input_image.shape[-2:] # 1024 1024
        original_image_size = image.shape[:2] # 512 512

        with torch.no_grad():
            # encode
            image_embedding = sam.image_encoder(input_image)

        # Iterate through all organs
        for k in range(1, 14):
            gt_mask = label == k
            if gt_mask.max() != 0:
                with torch.no_grad():
                    # get prompt
                    if args.number: # Use points as prompt
                        input_point, input_label = GetPointsFromMask(gt_mask, args.number, args.center)
                        input_point = transform.apply_coords(input_point, original_image_size)
                        input_point = torch.as_tensor(input_point, dtype=torch.float, device=device)
                        input_label = torch.as_tensor(input_label, dtype=torch.float, device=device)
                        input_point = input_point[None, :, :]
                        input_label = input_label[None, :]
                        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                            points = (input_point, input_label), #输入点和标签
                            boxes=None,
                            masks=None,
                        )
                    elif args.bbox: # Use bounding box as prompt
                        box = GetBBoxFromMask(gt_mask, True)
                        box = transform.apply_boxes(box, original_image_size)
                        box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
                        box_torch = box_torch[None, :]
                        
                        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                            points=None,
                            boxes=box_torch,
                            masks=None,
                        )
                        
                low_res_masks, iou_predictions = sam.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )

                upscaled_masks = sam.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)
                binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))
                gt_mask_resized = torch.from_numpy(np.resize(gt_mask, (1, 1, gt_mask.shape[0], gt_mask.shape[1]))).to(device)
                gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)

                if mode == 'train':
                    loss = loss_fn(upscaled_masks, gt_binary_mask)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss.append(loss.item())
                dice = 2 * torch.sum(binary_mask * gt_binary_mask) / (torch.sum(binary_mask) + torch.sum(gt_binary_mask))
                epoch_dice[k].append(dice.item())

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


plot_curve(losses, dices, dices_val)

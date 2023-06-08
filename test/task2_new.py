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

parser = argparse.ArgumentParser(description='Task 1')
parser.add_argument('-n', '--number', type=int,
                    help='Number of points to sample from mask')
parser.add_argument('-c', '--center',
                    help='Use center (max distance to boundary) of mask as the first prompt',
                    action='store_true')
parser.add_argument('-b', '--bbox',
                    help='Use bounding box of mask as prompt',
                    action='store_true')
parser.add_argument('-e', '--epoch',
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

# Get list of samples
train_path = 'BTCV/imagesTr'
label_path = 'BTCV/labelsTr'
train_list = os.listdir(train_path)
label_list = os.listdir(label_path)
train_list.sort()
label_list.sort()
print("Samples loaded")

# Helper functions

# Get points from mask
def GetPointsFromMask(mask, number):
    input_point = []
    input_label = []

    # If center flag is set, get center of mask
    if args.center:
        dist_img = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        x, y = np.unravel_index(dist_img.argmax(), dist_img.shape)
        input_point.append([y, x])
        input_label.append(1)
        number -= 1

    candidates = np.where(mask)
    for i in range(number):
        _ = np.random.randint(0, candidates[0].size)
        x = candidates[0][_]
        y = candidates[1][_]
        input_point.append([y, x])
        input_label.append(1)

    return np.array(input_point), np.array(input_label)

# Get bounding box in xyxy format from mask
def GetBBoxFromMask(mask):
    m = mask.nonzero()
    return np.array([m[1].min(), m[0].min(), m[1].max(), m[0].max()])

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
optimizer = torch.optim.AdamW(sam.mask_decoder.parameters(), lr=lr, weight_decay=wd)
# loss_fn = torch.nn.MSELoss()
loss_fn = my_dice_loss
# data augmentation
my_transform = tfs.Compose([
            tfs.RandomHorizontalFlip(p=0.5), 
            tfs.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3)
        ])

np.random.seed(0)
dices = np.zeros(13, dtype=np.float32)
times = np.zeros(13, dtype=np.float32)
epoch_num = args.epoch

losses = []
dices = []

for epoch in range(epoch_num):
    epoch_loss = []
    epoch_dice = {}
    for k in range(1, 14):
        epoch_dice[k] = []

    for i in tqdm(range(len(train_list))):
        img_3d = nib.load(os.path.join(train_path, train_list[i])).get_fdata()
        label_3d = nib.load(os.path.join(label_path, label_list[i])).get_fdata()

        # Iterate through all slices
        for j in tqdm(range(img_3d.shape[2])):
        # 只取中间的切片试试
        # mid = int(img_3d.shape[2] / 2)
        # for j in tqdm(range(mid, mid+2)):
            label = label_3d[:,:,j]
            if label.max() == 0:
                continue
            img = img_3d[:,:,j].copy()
            img -= img.min()
            img /= img.max()

            img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            input_image_torch = torch.as_tensor(img, device=device).permute(2, 0, 1).contiguous()
            # print(input_image_torch.size())
            
            # data_augmentation
            # input_image_torch = my_transform(input_image_torch)

            transformed_image = input_image_torch[None, :, :, :]#将图像转换成模型需要的格式
            input_image = sam.preprocess(transformed_image) # 问题:这个preprocess是干嘛的？
            # show_img(input_image, f'./test_img/test_preprocess.png')
            # print(input_image.size())
            # assert 1==0

            input_size = input_image.shape[-2:] # 1024 1024
            original_image_size = transformed_image.shape[-2:] # 512 512

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
                            input_point, input_label = GetPointsFromMask(gt_mask, args.number)
                            input_point = torch.as_tensor(input_point, dtype=torch.float, device=device)
                            input_label = torch.as_tensor(input_label, dtype=torch.float, device=device)
                            input_point = input_point[None, :, :]
                            input_label = input_label[None, :]
                            sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                                points = (input_point, input_label), #输入点和标签
                                masks=None,
                                boxes=None,
                            )
                        elif args.bbox: # Use bounding box as prompt
                            bbox = GetBBoxFromMask(gt_mask)
                            box_torch = torch.as_tensor(bbox, dtype=torch.float, device=device)
                            box_torch = box_torch.view(1,1,4)
                            # print(box_torch.size())
                            sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                                points=None,
                                masks=None,
                                boxes=box_torch,
                            )
                    low_res_masks, iou_predictions = sam.mask_decoder(
                        image_embeddings=image_embedding,
                        image_pe=sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )
                    low_res_masks = low_res_masks[..., : 128, : 128]

                    upscaled_masks = sam.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)
                    binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))
                    gt_mask_resized = torch.from_numpy(np.resize(gt_mask, (1, 1, gt_mask.shape[0], gt_mask.shape[1]))).to(device)
                    gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)

                    # show_mask(binary_mask, gt_binary_mask, input_point, k)
                    # show_mask_lrum(low_res_masks, upscaled_masks, k)


                    loss = loss_fn(binary_mask, gt_binary_mask)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss.append(loss.item())
                    dice = 2 * torch.sum(binary_mask * gt_binary_mask) / (torch.sum(binary_mask) + torch.sum(gt_binary_mask))
                    # print(f'dice: {dice.item()}')
                    epoch_dice[k].append(dice.item())

    print(f'Epoch:{epoch}')
    print(f'loss: {mean(epoch_loss)}')
    print(f'epoch_dice: {epoch_dice}')
    for k in range(1, 14):
        epoch_dice[k] = mean(epoch_dice[k])
    print(f'dice: {epoch_dice}')
    print(f'mean dice: {mean(epoch_dice.values())}')

    losses.append(mean(epoch_loss))
    dices.append(mean(epoch_dice.values()))

plot_curve(losses, dices)
    

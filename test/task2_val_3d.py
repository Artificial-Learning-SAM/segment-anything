import os
import argparse
import sys
sys.path.append("..")

import numpy as np
import nibabel as nib
import cv2
import matplotlib as mpl
import torch
import glob
from monai.losses import DiceCELoss

from segment_anything import sam_model_registry, SamPredictor
from data_utils import DataLoader, GetPointsFromMask, GetBBoxFromMask

parser = argparse.ArgumentParser(description='Task 2')
parser.add_argument('-p', '--prompt', type=str,
                    help='List of numbers. x>0 means sampling x points from mask. x<0 means sampling |x| points, but using center (max distance to boundary) of mask as the first point. x==0 means using bbox. E.g. "[0, 1, -1, 3]".')
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
sam_checkpoint = "../sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint, decoder_version="task2")
sam.to(device=args.device)
sam.eval()
print("SAM initialized")

# Start training
from statistics import mean
from torch.nn.functional import threshold, normalize
from tqdm import tqdm
import torchvision.transforms as tfs
from utils import *

weights = glob.glob('model/hybrid/epoch-*.pth')
weights.sort(key=lambda x: int(x.split('-')[1]))

np.random.seed(0)
dataloader_val = DataLoader('val', sam, args)

dices_val = []

def do_epoch(epoch, dataloader, mode):
    dices = {}
    for i in tqdm(range(len(dataloader))):
        image_embeddings, sparse_embeddings, dense_embeddings, gt_masks, organ, mask_to_nii = dataloader.get_batch()
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

        for j in range(len(organ)):
            pred = upscaled_masks[j] > 0
            gt = gt_binary_mask[j]
            a = 2 * torch.sum(pred * gt)
            b = torch.sum(pred) + torch.sum(gt)
            if (mask_to_nii[j], organ[j]) not in dices:
                dices[(mask_to_nii[j], organ[j])] = np.zeros(2)
            dices[(mask_to_nii[j], organ[j])][0] += a
            dices[(mask_to_nii[j], organ[j])][1] += b

    organ_dices = np.zeros(13)
    cnt = np.zeros(13)
    for k, v in dices.items():
        organ_dices[k[1] - 1] += v[0] / v[1]
        cnt[k[1] - 1] += 1
    organ_dices /= cnt
    return organ_dices

for w in weights:
    sam.mask_decoder.load_state_dict(torch.load(w))
    with torch.no_grad():
        organ_dices = np.zeros(13)
        for i in range(5):
            organ_dices += do_epoch(0, dataloader_val, 'val')
        organ_dices /= 5
        dices_val.append(organ_dices.mean())
    print(w)
    print(organ_dices)
    print(organ_dices.mean())

print('Best:', np.array(dices_val).argmax(), np.array(dices_val).max())

plt.figure(figsize=(10, 10))
plt.plot(list(range(len(dices_val))), dices_val)
plt.title(f'Mean epoch validation 3d dice')
plt.xlabel('Epoch Number')
plt.ylabel('dice')
plt.savefig(f'{args.prompt}_val.png')

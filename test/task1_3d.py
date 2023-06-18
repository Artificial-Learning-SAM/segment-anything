import os
import argparse
import sys
sys.path.append("..")

import torch
import numpy as np
import nibabel as nib
import cv2
import matplotlib as mpl
from tqdm import tqdm

from segment_anything import sam_model_registry, SamPredictor
from data_utils import DataLoader, GetPointsFromMask, GetBBoxFromMask

parser = argparse.ArgumentParser(description='Task 1')
parser.add_argument('-p', '--prompt', type=str,
                    help='List of numbers. x>0 means sampling x points from mask. x<0 means sampling |x| points, but using center (max distance to boundary) of mask as the first point. x==0 means using bbox. E.g. "[0, 1, -1, 3]".')
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

args.prompt = eval(args.prompt)

print("Imports done")

# Init SAM
sam_checkpoint = "../sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint, decoder_version="task2")
sam.to(device=args.device)
if args.decoder_weight is not None:
    sam.mask_decoder.load_state_dict(torch.load(args.decoder_weight))
sam.eval()
print("SAM initialized")

np.random.seed(0)
dataloader = DataLoader('test', sam, args)

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

    organ_dices = np.zeros((6, 13))
    cnt = np.zeros(13)
    for k, v in dices.items():
        organ_dices[k[0], k[1] - 1] = v[0] / v[1]
        cnt[k[1] - 1] += 1
    return organ_dices, cnt

with torch.no_grad():
    organ_dices = np.zeros((6, 13))
    cnt = np.zeros(13)
    for i in range(5):
        a, b = do_epoch(0, dataloader, 'test')
        organ_dices += a
        cnt += b
    for i in range(6):
        for j in range(13):
            print(f'{organ_dices[i, j] / 5:.4f}', end=' ')
        print()
    organ_dices = organ_dices.sum(axis=0) / cnt
print(organ_dices)
print(organ_dices.mean())
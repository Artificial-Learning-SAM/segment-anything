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
dices = np.zeros(13, dtype=np.float32)
count = np.zeros(13, dtype=np.float32)

with torch.no_grad():
    for i in tqdm(range(len(dataloader) * 5)):
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

        for j in range(len(organ)):
            pred = upscaled_masks[j] > 0
            gt = gt_binary_mask[j]
            dice = 2 * torch.sum(pred * gt) / (torch.sum(pred) + torch.sum(gt))
            dices[organ[j] - 1] += dice.item()
            count[organ[j] - 1] += 1

dices /= count
print('dices:', dices)
# print('count:', count)
print('average:', np.mean(dices))
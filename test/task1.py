import os
import argparse
import sys
sys.path.append("..")

import numpy as np
import nibabel as nib
import cv2
import matplotlib as mpl

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
args = parser.parse_args()

print("Imports done")

# Init SAM
sam_checkpoint = "../sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
print("SAM initialized")

np.random.seed(0)
dataloader = DataLoader('test')
dices = np.zeros(13, dtype=np.float32)
times = np.zeros(13, dtype=np.float32)

for i in range(dataloader.slice_num()):
    image, label = dataloader.get_slice()
    predictor.set_image(image)
    
    # Iterate through all organs
    for k in range(1, 14):
        mask = label == k
        if mask.max() != 0:
            if args.number: # Use points as prompt
                input_point, input_label = GetPointsFromMask(mask, args.number, args.center)
                masks, _, _ = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=False,
                )
            elif args.bbox: # Use bounding box as prompt
                bbox = GetBBoxFromMask(mask, random_shift=True)
                masks, _, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=bbox[None, :],
                    multimask_output=False,
                )


            dice = 2 * np.sum(masks[0] * mask) / (np.sum(masks[0]) + np.sum(mask))
            print(f"Slice {i}, Organ {k}, Dice {dice}")
            dices[k-1] += dice
            times[k-1] += 1

dices /= times
print('dices:', dices)
print('times:', times)
print('average:', np.mean(dices))
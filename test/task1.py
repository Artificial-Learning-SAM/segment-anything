import os
import argparse
import sys
sys.path.append("..")

import numpy as np
import nibabel as nib
import cv2
import matplotlib as mpl

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

# Get list of samples
test_path = 'BTCV/imagesTest'
label_path = 'BTCV/labelsTest'
test_list = os.listdir(test_path)
label_list = os.listdir(label_path)
test_list.sort()
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


np.random.seed(0)
dices = np.zeros(13, dtype=np.float32)
times = np.zeros(13, dtype=np.float32)

# Iterate through all 6 test samples
for i in range(len(test_list)):
    img_3d = nib.load(os.path.join(test_path, test_list[i])).get_fdata()
    label_3d = nib.load(os.path.join(label_path, label_list[i])).get_fdata()

    # Iterate through all slices
    for j in range(img_3d.shape[2]):
        label = label_3d[:,:,j]
        if label.max() == 0:
            continue
        img = img_3d[:,:,j].copy()
        img -= img.min()
        img /= img.max()

        img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        # colormap = mpl.colormaps['viridis']
        # colormap = mpl.colormaps['plasma']
        # img = (colormap(img)[:, :, :3] * 255).astype(np.uint8)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # cv2.imwrite(f"sample_{i+1}_slice_{j}_img.png", img)
        predictor.set_image(img)

        # Iterate through all organs
        for k in range(1, 14):
            mask = label == k
            if mask.max() != 0:
                if args.number: # Use points as prompt
                    input_point, input_label = GetPointsFromMask(mask, args.number)
                    masks, _, _ = predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        multimask_output=False,
                    )
                elif args.bbox: # Use bounding box as prompt
                    bbox = GetBBoxFromMask(mask)
                    masks, _, _ = predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=bbox[None, :],
                        multimask_output=False,
                    )


                dice = 2 * np.sum(masks[0] * (label == k)) / (np.sum(masks[0]) + np.sum((label == k)))
                print(f"Sample {i+1}, Slice {j}, Organ {k}, Dice {dice}")
                dices[k-1] += dice
                times[k-1] += 1
                
                # cv2.imwrite(f"sample_{i+1}_slice_{j}_organ_{k}_mask.png", masks[0] * 255)
                # label_k = (label == k) * 100
                # label_k[x, y] = 255
                # cv2.imwrite(f"sample_{i+1}_slice_{j}_organ_{k}_label.png", label_k)

        # Remove break to test all slices
        # break

    # Remove break to test all data
    # break

dices /= times
print('dices:', dices)
print('times:', times)
print('average:', np.mean(dices))
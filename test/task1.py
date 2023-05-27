import os
import sys
sys.path.append("..")

import numpy as np
import nibabel as nib
import cv2

from segment_anything import sam_model_registry, SamPredictor

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
train_path = 'BTCV/imagesTr'
label_path = 'BTCV/labelsTr'
train_list = os.listdir(train_path)
label_list = os.listdir(label_path)
train_list.sort()
label_list.sort()
print("Samples loaded")

dices = np.zeros(13, dtype=np.float32)
times = np.zeros(13, dtype=np.float32)

# Iterate through all 30 samples
for i in range(len(train_list)):
    img_3d = nib.load(os.path.join(train_path, train_list[i])).get_fdata()
    label_3d = nib.load(os.path.join(label_path, label_list[i])).get_fdata()

    # Iterate through all slices
    for j in range(img_3d.shape[2]):
        img = img_3d[:,:,j]
        img = (img / img.max() * 255).astype(np.uint8)
        label = label_3d[:,:,j]
        predictor.set_image(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
        # cv2.imwrite(f"sample_{i+1}_slice_{j}_img.png", img)

        # Iterate through all organs
        for k in range(1, 14):
            mask = np.where(label == k)
            if mask[0].size != 0:
                _ = np.random.randint(0, mask[0].size)
                x = mask[0][_]
                y = mask[1][_]
                input_point = np.array([[y, x]])
                input_label = np.array([1])

                masks, scores, logits = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
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
print(dices)
print(np.mean(dices))
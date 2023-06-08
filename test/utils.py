import torch
import numpy as np
import matplotlib.pyplot as plt

def my_dice_loss(pred_mask, gt_mask):
    dice = 2 * torch.sum(pred_mask * gt_mask) / (torch.sum(pred_mask) + torch.sum(gt_mask))
    loss = 1 - dice
    return loss

def show_mask(pred_mask, gt_mask, input_point, k):
    pred = pred_mask.detach().cpu().numpy()
    gt = gt_mask.detach().cpu().numpy()
    point = input_point.detach().cpu().numpy()
    point = point.reshape(point.shape[1],point.shape[2])
    print(point.shape)

    plt.figure(figsize=(100,100))

    plt.subplot(2,2,1)
    plt.title('point')
    bg = np.zeros((512,512))
    plt.imshow(bg)
    ax = plt.gca()
    ax.scatter(point[0][0], point[0][1], color='green', marker='*', s=5000, edgecolor='white', linewidth=1.25)

    plt.subplot(2,2,2)
    plt.title('pred')
    plt.imshow(pred.reshape(512,512))

    plt.subplot(2,2,3)
    plt.title('gt')
    plt.imshow(gt.reshape(512,512))

    plt.savefig(f'test{k}.png')

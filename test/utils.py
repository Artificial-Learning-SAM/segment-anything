import torch
import numpy as np
import matplotlib.pyplot as plt

def my_dice_loss(pred_mask, gt_mask):
    dice = 2 * torch.sum(pred_mask * gt_mask) / (torch.sum(pred_mask) + torch.sum(gt_mask))
    loss = 1 - dice
    return loss

def plot_curve(losses, dices, dices_val):

    plt.figure(figsize=(5,5))
    plt.plot(list(range(len(losses))), losses)
    plt.title('Mean epoch loss')
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.savefig('loss.png')

    plt.figure(figsize=(5,5))
    plt.plot(list(range(len(dices))), dices)
    plt.title('Mean epoch dice')
    plt.xlabel('Epoch Number')
    plt.ylabel('Dice')
    plt.savefig('dice.png')

    plt.figure(figsize=(5,5))
    plt.plot(list(range(len(dices_val))), dices_val)
    plt.title('Mean epoch validation dice')
    plt.xlabel('Epoch Number')
    plt.ylabel('Dice')
    plt.savefig('dice_val.png')
    

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

    plt.savefig(f'./test_img/test{k}.png')

def show_mask_lrum(low_res, upscaled_masks, k):
    um = upscaled_masks.detach().cpu().numpy()
    um = um.reshape((512, 512))
    lr = low_res.detach().cpu().numpy()
    lr = lr.reshape((256,256))

    plt.figure(figsize=(20,20))

    # plt.subplot(1,2,1)
    plt.title('lr')
    plt.imshow(lr)
    
    plt.subplot(1,2,2)
    plt.title('um')
    plt.imshow(um)

    plt.savefig(f'./test_img/test_lrum{k}.png')

def show_img(input_img, title, T = True):
    img = input_img.detach().cpu().numpy()
    img = img.reshape(img.shape[1], img.shape[2], -1)
    if T:
        img = img.T

    plt.figure(figsize=(20,20))

    # plt.subplot(1,2,1)
    plt.title('img')
    plt.imshow(img)
    

    # plt.subplot(1,2,2)
    # plt.title('um')
    # plt.imshow(um)

    plt.savefig(title)

    

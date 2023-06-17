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

from segment_anything import sam_model_registry, SamPredictor , SamAutomaticMaskGenerator
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
                    default="epoch-119-val-0.8411638965.pth")
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


#Init VGG
from cnn_network import vgg11_bn

# Init vgg
vgg = vgg11_bn()
vgg = vgg.to(device=args.device)
path_vgg = "./gt_epoch-19-val-0.5212053571.pth"
state = torch.load(path_vgg)
vgg.load_state_dict(state)
vgg.eval()
print("VGG initialized")

# from segment_anything.modeling.mask_decoder_3 import MaskDecoder3
# from segment_anything.modeling.transformer import TwoWayTransformer
# prompt_embed_dim = 256
# class_decoder=MaskDecoder3(
#             num_multimask_outputs=3,
#             transformer=TwoWayTransformer(
#                 depth=3,
#                 embedding_dim=prompt_embed_dim,
#                 mlp_dim=2048,
#                 num_heads=8,
#             ),
#             transformer_dim=prompt_embed_dim,
#             iou_head_depth=3,
#             iou_head_hidden_dim=256,
#         )

# class_decoder.to(device=args.device)
# state = torch.load("./epoch-6-val-0.68139.pth")
# class_decoder.load_state_dict(state)
# print("class_decoder initialized")

np.random.seed(0)
dataloader = DataLoader('test', sam, args)
mask_generator = SamAutomaticMaskGenerator(sam , pred_iou_thresh=0.5 ,box_nms_thresh= 0.7)

# def decoder_classifier(image , anns):
#     image = torch.from_numpy(image) #512*512
#     #transform into 1 * 3 * 512 * 512
#     image = image.unsqueeze(0).repeat(1,3,1,1)
#     #to float
#     image = image.float()
#     img_embedding = sam.image_encoder(sam.preprocess(image.to(device=args.device)))
#     from segment_anything.utils.transforms import ResizeLongestSide
#     resize = ResizeLongestSide(sam.image_encoder.img_size)
#     original_size = image.shape[-2:]
#     for ann in anns:
#         box = ann['bbox']
#         box = np.array(box)
#         box = resize.apply_boxes(box, original_size)
#         box_torch = torch.as_tensor(box, dtype=torch.float, device=args.device)
#         sparse, dense = sam.prompt_encoder(
#             points = None,
#             boxes = torch.stack([box_torch], dim=0),
#             masks = None,
#         )
#         #output device for all tensors
#         _,_,class_logits = class_decoder(
#             image_embeddings=img_embedding,
#             image_pe=sam.prompt_encoder.get_dense_pe(),
#             sparse_prompt_embeddings=sparse,
#             dense_prompt_embeddings=dense,
#             multimask_output=False,
#         )
#         vclass = torch.argmax(class_logits , dim=1).item()
#         ann['organ'] = vclass
#     # assert False
#     return anns
>>>>>>> cd104df69d60044ed91a524d6bf103d8366397ce
    

def cnn_classifier(image , anns):
    image = torch.from_numpy(image) 
    image = image / 255
    image = np.clip(image , 0 , 1)
    num = len(anns)
    #image : 512*512
    #masks : DICT
    #input : num * 2 * 512 * 512
    image = image.unsqueeze(0).repeat(num,1,1).unsqueeze(1)
    masks = torch.stack([torch.from_numpy(ann['segmentation']).unsqueeze(0) for ann in anns], dim=0)
    masks = masks.float()
    input = torch.cat((image , masks) , dim=1)
    input = input.float()
    input = input.to(device=args.device)
    print(input.max())
    print(input.min())
    for i in range(num):
        # print(input[i].unsqueeze(0).shape)
        # print(input[i])
        output = vgg(input[i].unsqueeze(0))
        # print(output)
        output_class = torch.argmax(output , dim=1)
        anns[i]['organ'] = output_class.item()
    return anns
    

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
cmap = mcolors.ListedColormap(['black', 'red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'white', 'orange', 'pink', 'purple', 'brown', 'gray', 'olive', 'lightblue', 'lightgreen'])

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    for ann in sorted_anns:
        segmentation = ann['segmentation'] #01 mask
        organ = ann['organ']
        #organ = rand in range(1,14)
        import random
        organ = random.randint(1,14)
        color = cmap(organ) # get color from cmap based on organ
        #不透明度改成0.5
        color = (color[0],color[1],color[2],0.5)
        output = np.zeros(segmentation.shape + (4,)) # (512,512,4)
        output[segmentation == 1] = color # set color to output
        ax.imshow(output)
        
    plt.axis('off')

def show_image_masks(image, masks, gt_mask,id):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(image,cmap='gray')
    show_anns(masks)
    for mask in masks:
        segmentation = mask['segmentation']
        organ = mask['organ']
        y, x = np.where(segmentation)
        x_mean = np.mean(x).astype(int)
        y_mean = np.mean(y).astype(int)
        plt.text(x_mean, y_mean, organ, fontsize=8, color='red', ha='center', va='center')
    plt.axis('off')
    
    plt.subplot(1,2,2)
    # gt masks : (512,512) , each of them is 0 to 14 , 0 is background , i is one of the 13 organs
    # I want to show the gt_mask with different color
    unique_elements = np.unique(gt_mask)
    print("Number of unique elements in gt_mask:", len(unique_elements))

    #map 14 to 255
    
    plt.imshow(image, cmap='gray')
    output = np.zeros(gt_mask.shape + (4,))
    for i in range(1,14):
        if np.sum(gt_mask == i) == 0:
            continue
        color = cmap(i)
        color = (color[0],color[1],color[2],1)
        output[gt_mask == i] = color
        
        segmentation = gt_mask == i
        organ = i
        y, x = np.where(segmentation)
        x_mean = np.mean(x).astype(int)
        y_mean = np.mean(y).astype(int)
        # print(x_mean, y_mean, organ)
        plt.text(x_mean, y_mean, organ, fontsize=8, color='red', ha='center', va='center')

    plt.imshow(output)
    plt.axis('off')
    
    plt.savefig(f'test_myoutput{str(id)}.png')


def delete_multiple_masks(masks):
    # Create a dictionary to store the masks with the largest area for each organ
    organ_masks = {}
    # for mask in masks:
    #     organ = mask['organ']
    #     area = mask['area']
    #     if organ == 0:
    #         continue
    #     if organ not in organ_masks or area > organ_masks[organ]['area']:
    #         organ_masks[organ] = mask
    # Convert the dictionary back to a list
    # return list(organ_masks.values())
    print('find masks with length',len(masks))
    return masks


def calc_dice(pred_mask, gt_mask):
    # Calculate the mean dice score for a single organ
    intersection = np.logical_and(pred_mask, gt_mask)
    return 2 * intersection.sum() / (pred_mask.sum() + gt_mask.sum())
def calc_all_scores(masks , gt_mask):
    '''
    masks as a list of dict , dict has key 'segmentation' , which is a 512*512 np array
    dict has key 'organ' , which is a int
    gt_mask is a 512*512 np array
    value of gt_mask is 0 to 14 , 0 is background , i is one of the 13 organs
    '''
    pass


vgg.eval()
with torch.no_grad():
    for i in range(len(dataloader)):
        image, gt_mask = dataloader.get_single_image()
        #image : (512,512,3)
        #gt_mask : (512,512)
        pred_masks = mask_generator.generate(image)
        # print(len(pred_masks))
        # print(pred_masks[0].keys())
        count_masks = len(pred_masks)
        image = image[:,:,0]
        pred_masks = cnn_classifier(image , pred_masks)
        # pred_masks = decoder_classifier(image , pred_masks)
        pred_masks = delete_multiple_masks(pred_masks)
        show_image_masks(image , pred_masks,gt_mask ,i)
        #the number of non-zero element in gt_mask , multipky count only once
        #gt_mask: (512,512)
        #gt_masks: (num_gt,512,512)
        #gt_masks
        # show_masks_with_class(image , pred_masks , pred_organ_label)

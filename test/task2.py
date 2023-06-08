
import os
import argparse
import torch
import sys
sys.path.append("..")

import numpy as np 
import nibabel as nib
import cv2

from segment_anything import sam_model_registry, SamPredictor

parser = argparse.ArgumentParser(description='Task 2')
parser.add_argument('-n', '--number', type=int,
                    help='Number of points to sample from mask')
parser.add_argument('-c', '--center',
                    help='Use center (max distance to boundary) of mask as the first prompt',
                    action='store_true')
parser.add_argument('-b', '--bbox',
                    help='Use bounding box of mask as prompt',
                    action='store_true')
args = parser.parse_args()

args = parser.parse_args()

# print("Imports done")

# # Init SAM
sam_checkpoint = "../sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
sam.train()
# predictor = SamPredictor(sam)
print("SAM initialized")

# Get list of samples
train_path = 'BTCV/imagesTr'
label_path = 'BTCV/labelsTr'
train_list = os.listdir(train_path)
label_list = os.listdir(label_path)
train_list.sort()
label_list.sort()
train_list = train_list[:1]
label_list = label_list[:1]

# optimizer = torch.optim.Adam(sam.mask_decoder.parameters()) 

loss_fn = torch.nn.MSELoss()



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




from segment_anything.utils.transforms import ResizeLongestSide
from collections import defaultdict
transformed_data = defaultdict(dict)

number_of_trained_data = 0
for i in range(len(train_list)):
    img_3d = nib.load(os.path.join(train_path, train_list[i])).get_fdata()
    label_3d = nib.load(os.path.join(label_path, label_list[i])).get_fdata()
    print(img_3d.shape)
    print(label_3d.shape)
    
    # Iterate through all slices
    for j in range(img_3d.shape[2]):
        img = img_3d[:,:,j]
        img = (img / img.max() * 255).astype(np.uint8)
        label = label_3d[:,:,j]
        for k in range(1, 14):
            gt_mask = label == k

            #####直接用preprocess扩成三通道
            # input_point, input_label = GetPointsFromMask(gt_mask, 1)
            # transform = ResizeLongestSide(sam.image_encoder.img_size)#ResizeLongestSide是一个自定义的变换类，作用是将图像调整为具有指定最长边长度的目标大小。
            # input_image = transform.apply_image(img)#变换图像大小
            # input_image_torch = torch.as_tensor(input_image, device=device)#将调整过大小的图像转换成张量对象，分配给设备
            # transformed_image = input_image_torch.contiguous()[None, None, :, :]#将图像转换成模型需要的格式
            # input_image = sam.preprocess(transformed_image)
            #print("img_size = {}".format(img.shape))
            #print("imput_image_size = {}".format(input_image.shape))

            #####用cv2.COLOR_GRAY2RGB扩成三通道
            input_image=cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            input_image_torch = torch.as_tensor(input_image, device=device)
            print("input_image_size after GRAY2RGB = {}".format(input_image.shape))
            transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]#将图像转换成模型需要的格式
            input_image = sam.preprocess(transformed_image)
            print("input_image_size after transform = {}".format(transformed_image.shape))
            print("input_image_size after preprocess = {}".format(input_image.shape))

            original_image_size = img.shape[:2]  #512 512
            input_size = input_image.shape[-2:]  #1024 1024
            if gt_mask.max() == 0: continue
            #往transformed_data中添加数据  
            # print(number_of_trained_data)
            transformed_data[number_of_trained_data]["input_image"] = input_image  #1 3 1024 1024
            transformed_data[number_of_trained_data]["input_size"] = input_size  #1024 1024
            transformed_data[number_of_trained_data]["original_image_size"] = original_image_size  #512 512
            transformed_data[number_of_trained_data]["ground_truth_mask"] = gt_mask  #512 512
            number_of_trained_data += 1

print (number_of_trained_data)
print(transformed_data[0]["input_image"].shape)
print(transformed_data[0]["input_size"])
print(transformed_data[0]["original_image_size"])
print(transformed_data[0]["ground_truth_mask"].shape)


#from here end of the process


lr = 1e-5
wd = 0
# optimizer = torch.optim.Adam(sam.mask_decoder.parameters(), lr=lr, weight_decay=wd)
# 使用自适应的学习率
optimizer = torch.optim.AdamW(sam.mask_decoder.parameters(), lr=lr, weight_decay=wd)

loss_fn = torch.nn.MSELoss()
# loss_fn = torch.nn.BCELoss()
keys = list(transformed_data.keys())


def calc_dice(transformed_data, sam):
    input_image = transformed_data['input_image'].to(device) #1 3 1024 1024
    input_size = transformed_data['input_size'] #1024 1024
    original_image_size = transformed_data['original_image_size'] #512 512
    ground_truth_mask = transformed_data['ground_truth_mask'] #512 512
    input_point, input_label = GetPointsFromMask(ground_truth_mask, 3)
    predictor = SamPredictor(sam)
    #ValueError: pic should be 2/3 dimensional. Got 4 dimensions.
    predictor.set_image(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    #print("SET_IMGE_SIZE = {}".format(input_image[0].shape))
    #print("INPUT_POINT_SIZE = {}".format(input_point.shape))
    #print("INPUT_LABEL_SIZE = {}".format(input_label.shape))
    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    #计算预测的dice
    #print("masks.shape = {}".format(masks[0].shape))
    #print("ground_truth_mask.shape = {}".format(ground_truth_mask.shape))
    #dice = 2 * np.sum(masks[0] * (label == k)) / (np.sum(masks[0]) + np.sum((label == k)))
    dice = 2 * np.sum(masks[0] * ground_truth_mask) / (np.sum(masks[0]) + np.sum(ground_truth_mask))
    print(f'Dice: {dice}')





#Run fine tuning
from statistics import mean

from torch.nn.functional import threshold, normalize

num_epochs = 100
losses = []

for epoch in range(num_epochs):
    epoch_losses = []
    # Just train on the first 20 examples
    for k in keys[:20]:
        input_image = transformed_data[k]['input_image'].to(device)
        input_size = transformed_data[k]['input_size']
        original_image_size = transformed_data[k]['original_image_size']
        ground_truth_mask = transformed_data[k]['ground_truth_mask']
        if ground_truth_mask.max() == 0: continue
        # print(k, input_image.shape, input_size, original_image_size, ground_truth_mask.shape)
        # No grad here as we don't want to optimise the encoders
        with torch.no_grad():
            image_embedding = sam.image_encoder(input_image)#对输入图像编码
            
            input_point, input_label = GetPointsFromMask(ground_truth_mask, 3)
            input_point = torch.as_tensor(input_point, dtype=torch.float, device=device)
            input_label = torch.as_tensor(input_label, dtype=torch.float, device=device)
            input_point = input_point[None, :, :]
            input_label = input_label[None, :]
            # print(input_point.shape)
        #   prompt_box = bbox_coords[k]#从bbox_coords字典中获取键为k的边界框坐标
        #   box = transform.apply_boxes(prompt_box, original_image_size)#利用transform对象，将边界框坐标prompt_box应用到原始图像大小original_image_size上，生成变换后的边界框坐标box。
        #   box_torch = torch.as_tensor(box, dtype=torch.float, device=device)#将变换后的边界框坐标box转换为PyTorch张量对象，并将张量分配给指定设备
        #   box_torch = box_torch[None, :]#对box_torch张量进行了维度扩展操作，在最前面增加了一个额外的维度。原始的box_torch张量的形状可能是 (N, 4)，其中 N 是边界框的数量，4 是边界框的坐标维度。通过添加一个额外的维度，形状将变为 (1, N, 4)。这种维度扩展通常用于将单个样本的边界框转换为批处理形式，其中第一个维度表示批处理大小。在这里，通过添加一个额外的维度，表示只有一个样本，并且批处理大小为1。这样，box_torch张量将被扩展为 (1, N, 4) 的形状，并可以用于后续的模型输入或处理。
        
            sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                points = (input_point, input_label),#输入点和标签
                masks=None,
                boxes=None,
            )
        low_res_masks, iou_predictions = sam.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        upscaled_masks = sam.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)
        binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))

        gt_mask_resized = torch.from_numpy(np.resize(ground_truth_mask, (1, 1, ground_truth_mask.shape[0], ground_truth_mask.shape[1]))).to(device)
        gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)
        
        loss = loss_fn(binary_mask, gt_binary_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
    losses.append(epoch_losses)
    print(f'EPOCH: {epoch}')
    print(f'Mean loss: {mean(epoch_losses)}')
    calc_dice(transformed_data[keys[-1]], sam)

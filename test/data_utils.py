import os

import cv2
import torch
import random
import numpy as np
import nibabel as nib
import matplotlib as mpl
import psutil
from tqdm import tqdm

from segment_anything.utils.transforms import ResizeLongestSide

class DataLoader:
    def __init__(self, mode, sam, args, get_cnn=False, aug=True , single_nii=False):
        print(f'Loading {mode} data...')
        self.mode = mode
        self.sam = sam
        self.args = args
        self.idx = 0
        self.get_cnn = get_cnn
        self.aug = aug

        # Get file list
        if mode == 'train' or mode == 'val':
            image_dir = 'BTCV/imagesTr'
            label_dir = 'BTCV/labelsTr'
        elif mode == 'test':
            image_dir = 'BTCV/imagesTs'
            label_dir = 'BTCV/labelsTs'
        image_list = os.listdir(image_dir)
        label_list = os.listdir(label_dir)
        image_list.sort()
        label_list.sort()
        if single_nii:
            image_list = image_list[:1]
            label_list = label_list[:1]        

        # Split train and val
        if mode == 'train':
            image_list = image_list[:len(image_list) // 6 * 5]
            label_list = label_list[:len(label_list) // 6 * 5]
        elif mode == 'val':
            image_list = image_list[len(image_list) // 6 * 5:]
            label_list = label_list[len(label_list) // 6 * 5:]

        # Data augmentation
        if mode == 'train':
            self.transforms = [
                lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE),
                lambda x: cv2.rotate(x, cv2.ROTATE_180),
                lambda x: cv2.flip(x, 0)
            ]

        # Get every slice
        slices = []
        labels = []
        
        for i in range(len(image_list)):
            assert image_list[i][-11:] == label_list[i][-11:]
            image_path = os.path.join(image_dir, image_list[i])
            label_path = os.path.join(label_dir, label_list[i])
            image_3d = nib.load(image_path).get_fdata()
            label_3d = nib.load(label_path).get_fdata()

            # Perform ScaleIntensityRange
            a_min = -175
            a_max = 250
            image_3d = (image_3d - a_min) / (a_max - a_min)
            image_3d = np.clip(image_3d, 0, 1)
            
            for j in range(image_3d.shape[2]):
                image = image_3d[:, :, j]
                label = label_3d[:, :, j]
                if label.max() == 0:
                    continue

                image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
                # colormap = mpl.colormaps['viridis']
                # image = (colormap(image)[:, :, :3] * 255).astype(np.uint8)

                if mode == 'train' and self.aug:
                    slices += self.augment(image)
                    labels += self.augment(label)
                else:
                    slices.append(image)
                    labels.append(label)

        self.slices = slices
        self.labels = labels
        self.original_size = slices[0].shape[:2]
        self.input_size = (sam.image_encoder.img_size, sam.image_encoder.img_size)
        self.resize = ResizeLongestSide(sam.image_encoder.img_size)


        # Get every mask
        self.masks = []
        self.masks_idx = []
        self.organ = []
        for i in range(len(labels)):
            # Add background
            if self.get_cnn:
                lb = 0
            else:
                lb = 1
            
            for j in range(lb, 14):
                mask = (labels[i] == j).astype(np.uint8)
                if mask.max() > 0:
                    self.masks.append(mask)
                    self.masks_idx.append(i)
                    self.organ.append(j)
        self.len = len(self.masks)
        self.len = self.len // args.batch_size * args.batch_size

        print(f'Loaded {mode} data.')
        print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3, 'GB')

    def augment(self, image):
        """
        Perform data augmentation.

        Args:
            image (np.ndarray): Image to be augmented.

        Returns:
            list: Augmented images.
        """
        images = [image]
        for transform in self.transforms:
            for i in range(len(images)):
                images.append(transform(images[i]))
        assert len(images) == 2**len(self.transforms)
        return images
    
    def get_image_embeddings(self, idxes):
        """
        Get image embeddings.

        Args:
            idxes (list): Indexes of the images.

        Returns:
            torch.Tensor: Image embeddings.
        """
        image_embeddings = []
        for i in idxes:
            if self.aug:
                dir = 'embeddings'
            else:
                dir = 'embeddings_noaug'
            path = f'{dir}/{self.mode}/{i}.pt'
            if os.path.exists(path):
                image_embeddings.append(torch.load(path))
            else:
                with torch.no_grad():
                    image = self.resize.apply_image(self.slices[i])
                    image = torch.as_tensor(image, device=self.args.device)
                    image = image.permute(2, 0, 1).contiguous()
                    image = image[None, :, :, :]
                    image = self.sam.image_encoder(self.sam.preprocess(image))
                image_embeddings.append(image)
                if not os.path.exists(f'{dir}'):
                    os.mkdir(f'{dir}')
                if not os.path.exists(f'{dir}/{self.mode}'):
                    os.mkdir(f'{dir}/{self.mode}')
                torch.save(image, path)

        image_embeddings = torch.cat(image_embeddings, dim=0)
        return image_embeddings

    def get_single_image(self):
        """
        Get a single image , gt_masks

        Returns:
        """
        image = self.slices[self.idx]
        gt_masks = self.labels[self.idx]
        self.idx += 5
        if self.idx == self.len:
            self.idx = 0
        return image, gt_masks

    def get_batch(self):
        """
        Get the next batch of the dataset.

        Returns:
        """
        image_embeddings = self.get_image_embeddings(
            self.masks_idx[self.idx : self.idx + self.args.batch_size]
        )
        i = self.masks_idx[self.idx : self.idx + self.args.batch_size]
        masks = self.masks[self.idx : self.idx + self.args.batch_size]
        organ = self.organ[self.idx : self.idx + self.args.batch_size]

        if self.get_cnn:
            img = np.array([self.slices[_][:,:,0] / 255 for _ in i])
            img = torch.as_tensor(img, device=self.args.device)

        self.idx += self.args.batch_size
        if self.idx == self.len:
            self.idx = 0

        # Get the prompt embeddings
        with torch.no_grad():
            types = len(self.args.prompt)
            inputs, sparse, dense = {}, {}, {}
            for i in self.args.prompt:
                if i == 0:
                    inputs[i] = []
                else:
                    inputs[i] = [[], []]

            # Preprocess the prompts
            prompt_types = []
            for i in range(len(masks)):
                mask = masks[i]
                prompt_type = random.choice(self.args.prompt)
                prompt_types.append(prompt_type)
                if prompt_type == 0: # Use bbox as prompt
                    box = GetBBoxFromMask(mask, True)
                    box = self.resize.apply_boxes(box, self.original_size)
                    box_torch = torch.as_tensor(box, dtype=torch.float, device=self.args.device)
                    inputs[0].append(box_torch)
                else: # Use points as prompt
                    input_point, input_label = GetPointsFromMask(
                        mask, abs(prompt_type), prompt_type < 0
                    )
                    input_point = self.resize.apply_coords(input_point, self.original_size)
                    input_point = torch.as_tensor(input_point, dtype=torch.float, device=self.args.device)
                    input_label = torch.as_tensor(input_label, dtype=torch.float, device=self.args.device)
                    inputs[prompt_type][0].append(input_point)
                    inputs[prompt_type][1].append(input_label)
                    
            # Each type of prompts are encoded as a batch
            for i in self.args.prompt:
                if i == 0:
                    sparse[i], dense[i] = self.sam.prompt_encoder(
                        points = None,
                        boxes = torch.stack(inputs[0]),
                        masks = None,
                    )
                else:
                    sparse[i], dense[i] = self.sam.prompt_encoder(
                        points = (torch.stack(inputs[i][0]), torch.stack(inputs[i][1])),
                        boxes = None,
                        masks = None,
                    )
            
            # Get the embeddings
            sparse_embeddings = []
            dense_embeddings = []
            for i in range(len(masks)):
                sparse_embeddings.append(sparse[prompt_types[i]][0])
                sparse[prompt_types[i]] = sparse[prompt_types[i]][1:]
                dense_embeddings.append(dense[prompt_types[i]][0])
                dense[prompt_types[i]] = dense[prompt_types[i]][1:]
            sparse_embeddings = torch.stack(sparse_embeddings)
            dense_embeddings = torch.stack(dense_embeddings)

        masks = np.array(masks)
        masks = torch.as_tensor(masks, dtype=torch.float, device=self.args.device)
        if self.get_cnn:
            return image_embeddings, sparse_embeddings, dense_embeddings, masks, organ, img
        return image_embeddings, sparse_embeddings, dense_embeddings, masks, organ

    def __len__(self):
        """
        Get number of batches in dataset.
        """
        return self.len // self.args.batch_size


def GetPointsFromMask(mask, number, include_center):
    """
    Get points from mask.

    Arguments:
        mask: (H, W)
        number: number of points to get
        include_center: whether to include center of mask
    """
    input_point = []
    input_label = []

    # If center flag is set, get center of mask
    if include_center:
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


def GetBBoxFromMask(mask, random_shift):
    """
    Get bounding box in xyxy format from mask.

    Arguments:
        mask: (H, W)
        random_shift: whether to randomly shift the bounding box
    """
    m = mask.nonzero()
    bbox = np.array([m[1].min(), m[0].min(), m[1].max(), m[0].max()])

    if random_shift:
        bbox[0] -= np.random.randint(0, 10)
        bbox[1] -= np.random.randint(0, 10)
        bbox[2] += np.random.randint(0, 10)
        bbox[3] += np.random.randint(0, 10)

        # Clip bounding box to image
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, mask.shape[1] - 1)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, mask.shape[0] - 1)

    return bbox

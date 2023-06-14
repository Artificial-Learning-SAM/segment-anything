import os

import cv2
import torch
import numpy as np
import nibabel as nib
import matplotlib as mpl
import psutil
from tqdm import tqdm

from segment_anything.utils.transforms import ResizeLongestSide

class DataLoader:
    def __init__(self, mode, sam, args):
        print(f'Loading {mode} data...')
        self.sam = sam
        self.args = args
        self.idx = 0

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

                if mode == 'train':
                    slices += self.augment(image)
                    labels += self.augment(label)
                else:
                    slices.append(image)
                    labels.append(label)

        self.original_size = slices[0].shape[:2]
        self.input_size = (sam.image_encoder.img_size, sam.image_encoder.img_size)

        # Preprocess image embeddings
        self.image_embeddings = []
        self.resize = ResizeLongestSide(sam.image_encoder.img_size)
        with torch.no_grad():
            print('Preprocessing image embeddings...')
            for i in tqdm(range(len(slices))):
                image = self.resize.apply_image(slices[i])
                image = torch.as_tensor(image, device=args.device)
                image = image.permute(2, 0, 1).contiguous()
                image = image[None, :, :, :]
                self.image_embeddings.append(sam.image_encoder(sam.preprocess(image)))

        # Get every mask
        self.masks = []
        self.masks_idx = []
        self.organ = []
        for i in range(len(labels)):
            for j in range(1, 14):
                mask = (labels[i] == j).astype(np.uint8)
                if mask.max() > 0:
                    self.masks.append(mask)
                    self.masks_idx.append(i)
                    self.organ.append(j)
        self.len = len(self.masks)
        self.len = self.len // args.batch_size * args.batch_size

        print(f'Loaded {mode} data.')

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

    def get_batch(self):
        """
        Get the next batch of the dataset.

        Returns:
        """
        i = self.masks_idx[self.idx : self.idx + self.args.batch_size]
        image_embeddings = [self.image_embeddings[_] for _ in i]
        image_embeddings = torch.cat(image_embeddings, dim=0)
        masks = self.masks[self.idx : self.idx + self.args.batch_size]
        organ = self.organ[self.idx : self.idx + self.args.batch_size]
        self.idx += self.args.batch_size
        if self.idx == self.len:
            self.idx = 0

        with torch.no_grad():
            if self.args.number: # Use points as prompt
                input_points, input_labels = [], []
                for mask in masks:
                    input_point, input_label = GetPointsFromMask(mask, self.args.number, self.args.center)
                    input_point = self.resize.apply_coords(input_point, self.original_size)
                    input_point = torch.as_tensor(input_point, dtype=torch.float, device=self.args.device)
                    input_label = torch.as_tensor(input_label, dtype=torch.float, device=self.args.device)
                    input_points.append(input_point)
                    input_labels.append(input_label)
                    
                sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                    points = (torch.stack(input_points), torch.stack(input_labels)),
                    boxes = None,
                    masks = None,
                )
            elif self.args.bbox: # Use bounding box as prompt
                boxes = []
                for mask in masks:
                    box = GetBBoxFromMask(mask, True)
                    box = self.resize.apply_boxes(box, self.original_size)
                    box_torch = torch.as_tensor(box, dtype=torch.float, device=self.args.device)
                    boxes.append(box_torch)
                
                sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                    points = None,
                    boxes = torch.stack(boxes),
                    masks = None,
                )

        masks = np.array(masks)
        masks = torch.as_tensor(masks, dtype=torch.float, device=self.args.device)
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

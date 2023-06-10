import os

import cv2
import numpy as np
import nibabel as nib

class DataLoader:
    def __init__(self, mode):
        self.mode = mode
        self.image_idx = 0
        self.slice_idx = 0

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

        self.images_3d = []
        self.labels_3d = []
        for i in range(len(image_list)):
            assert image_list[i][-11:] == label_list[i][-11:]
            image_path = os.path.join(image_dir, image_list[i])
            label_path = os.path.join(label_dir, label_list[i])
            image_3d = nib.load(image_path).get_fdata()
            label_3d = nib.load(label_path).get_fdata()
            image_3d -= image_3d.min()
            image_3d /= image_3d.max()
            image_slices = []
            label_slices = []
            for j in range(image_3d.shape[2]):
                image = image_3d[:, :, j]
                label = label_3d[:, :, j]
                if label.max() == 0:
                    continue

                image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
                # colormap = mpl.colormaps['viridis']
                # img = (colormap(img)[:, :, :3] * 255).astype(np.uint8)

                image_slices.append(image)
                label_slices.append(label)

            image_3d = np.stack(image_slices, axis=0)
            label_3d = np.stack(label_slices, axis=0)

            self.images_3d.append(image_3d)
            self.labels_3d.append(label_3d)

    def get_slice(self):
        """
        Get the next slice of the dataset.

        Returns:
            image: (H, W, 3)
            label: (H, W)
        """
        image = self.images_3d[self.image_idx][self.slice_idx]
        label = self.labels_3d[self.image_idx][self.slice_idx]
        self.slice_idx += 1
        if self.slice_idx == self.images_3d[self.image_idx].shape[0]:
            self.image_idx += 1
            self.slice_idx = 0
        if self.image_idx == len(self.images_3d):
            self.image_idx = 0
        return image, label

    def slice_num(self):
        """
        Get number of slices in dataset.
        """
        num = 0
        for image_3d in self.images_3d:
            num += image_3d.shape[0]
        return num


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

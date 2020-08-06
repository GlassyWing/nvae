import os
from glob import glob

import cv2
import h5py
import torch
from torch.utils.data import Dataset


class ImageH5Dataset(Dataset):

    def __init__(self, dataset_path, img_dim):
        self.dataset_path = dataset_path
        self.img_dim = (img_dim, img_dim) if type(img_dim) == int else img_dim
        self.live_data = None
        self.images = None

        with h5py.File(self.dataset_path, 'r') as file:
            self.dataset_len = len(file["image"])

    def __getitem__(self, idx):
        if self.live_data is None:
            self.live_data = h5py.File(self.dataset_path, 'r')
            self.images = self.live_data['image']

        image = self.images[idx]

        h, w, c = image.shape
        top_h = int((h - w) / 2)

        image = image[top_h:top_h + w]
        image = cv2.resize(image, self.img_dim, interpolation=cv2.INTER_LINEAR)
        image = image / 255.

        return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

    def __len__(self):
        return self.dataset_len


class ImageFolderDataset(Dataset):

    def __init__(self, image_dir, img_dim):
        self.img_paths = glob(os.path.join(image_dir, "*.jpg"))
        self.img_dim = (img_dim, img_dim) if type(img_dim) == int else img_dim

    def __getitem__(self, idx):
        image = cv2.imread(self.img_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w, c = image.shape
        if h > w:
            top_h = int((h - w) / 2)
            image = image[top_h:top_h + w]
        else:
            left_w = int((w - h) / 2)
            image = image[:, left_w:left_w + h]
        image = cv2.resize(image, self.img_dim, interpolation=cv2.INTER_LINEAR)
        image = image / 255.

        return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

    def __len__(self):
        return len(self.img_paths)

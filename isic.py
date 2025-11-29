import os
from torch.utils.data.dataset import Dataset
import torch
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
transforms_ = {
    'train': [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ],
    'test': [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]
}


def get_isic(data_path, split="train", preprocessing=True):
    if split == "train":
        transform = transforms_['train']
    else:
        transform = transforms_['test']
    if not preprocessing:
        transform = transforms.Compose([t for t in transform if not isinstance(t, transforms.Normalize)])
    else:
        transform = transforms.Compose(transform)
    return ImageNetDataset(data_path, train=split == "train", transform=transform)


class ImageNetDataset(datasets.ImageFolder):
    def __init__(self, root, *args, validate=False, train=True, use_precomputed_labels=False,
                 labels_path=None, transform=None, **kwargs):
        """ImageNet root folder is expected to have two directories: train and test."""

        if train and validate == train:
            raise ValueError('Train and validate can not be True at the same time.')
        if use_precomputed_labels and labels_path is None:
            raise ValueError('If use_precomputed_labels=True the labels_path is necessary.')

        if train:
            root = os.path.join(root, 'train')
        #elif validate:
        #    root = os.path.join(root, 'val_mpeg')
        else:
            root = os.path.join(root, 'test')

        super().__init__(root, transform=transform, *args, **kwargs)
        self.transforms = transform
        self.preprocessing = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if validate and use_precomputed_labels:
            df = pd.read_csv(labels_path, sep='\t')
            df.input_path = df.input_path.apply(lambda x: os.path.join(root, x))
            mapping = dict(zip(df.input_path, df.pred_class))

            self.samples = [(x[0], mapping[x[0]]) for x in self.samples]
            self.targets = [x[1] for x in self.samples]

        self.class_names = (list(CLASS_NAMES.values()))
        self.num_classes = len(self.class_names)

        #self.bird_class_ids = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 
        #                       80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 
        #                       99, 97, 98, 100, 127, 128, 129, 130, 132, 131, 133, 134, 135, 137, 138, 
        #                       139, 140, 141, 142, 143, 136, 144, 145, 146]


    def reverse_augmentation(self, data: torch.Tensor) -> torch.Tensor:
        data = data.clone() + 0
        mean = torch.Tensor(self.preprocessing.mean).to(data)
        var = torch.Tensor(self.preprocessing.std).to(data)
        data *= var[:, None, None]
        data += mean[:, None, None]
        return torch.multiply(data, 255)

    @staticmethod
    def class_id_name(class_id: int) -> str:
        return CLASS_NAMES[class_id]

    def get_target(self, index):
        return self.targets[index]
    

CLASS_NAMES = {'0': 'benign',
               '1': 'malignant'}

class ISICDataset(Dataset):
    def __init__(self, basedir, split="train", transform=None):
        try:
            split_i = ["train", "val", "test"].index(split)
        except ValueError:
            raise(f"Unknown split {split}")
        #metadata_df = pd.read_csv(os.path.join(basedir, "metadata.csv"))
        #metadata_df = pd.read_csv("./data/meta_4groups.csv")
        metadata_df = pd.read_csv("raw_val_4groups.csv")
        #metadata_df = pd.read_csv(os.path.join(basedir, "mask_overlays_metadata.csv"))
        self.metadata_df = metadata_df[metadata_df["split"] == split_i]
        self.basedir = basedir
        self.transform = transform
        self.y_array = self.metadata_df['benign_malignant'].values
        self.p_array = self.metadata_df['patches'].values
        self.n_classes = np.unique(self.y_array).size
        self.confounder_array = self.metadata_df['patches'].values
        self.n_places = np.unique(self.confounder_array).size
        self.group_array = (self.y_array * self.n_places + self.confounder_array).astype('int')
        self.n_groups = self.n_classes * self.n_places
        self.group_counts = (
                torch.arange(self.n_groups).unsqueeze(1) == torch.from_numpy(self.group_array)).sum(1).float()
        self.y_counts = (
                torch.arange(self.n_classes).unsqueeze(1) == torch.from_numpy(self.y_array)).sum(1).float()
        self.p_counts = (
                torch.arange(self.n_places).unsqueeze(1) == torch.from_numpy(self.p_array)).sum(1).float()
        self.filename_array = self.metadata_df['isic_id'].values

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]
        p = self.confounder_array[idx]
        n = self.filename_array[idx]

        img_path = os.path.join(self.basedir, self.filename_array[idx])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, y, g, p, n

class MaskedISIC(Dataset):
    def __init__(self, basedir, split="train", transform=None):
        try:
            split_i = ["train", "val", "test"].index(split)
        except ValueError:
            raise(f"Unknown split {split}")
        #metadata_df = pd.read_csv(os.path.join(basedir, "metadata.csv"))
        #metadata_df = pd.read_csv("./data/meta_4groups.csv")
        #metadata_df = pd.read_csv("/home/leph/data/isic/raw_val_4groups.csv")
        metadata_df = pd.read_csv(os.path.join(basedir, "mask_overlays_metadata.csv"))
        self.metadata_df = metadata_df[metadata_df["split"] == split_i]
        self.basedir = basedir
        self.transform = transform
        self.y_array = self.metadata_df['benign_malignant'].values
        self.p_array = self.metadata_df['patches'].values
        self.n_classes = np.unique(self.y_array).size
        self.confounder_array = self.metadata_df['patches'].values
        self.n_places = np.unique(self.confounder_array).size
        self.group_array = (self.y_array * self.n_places + self.confounder_array).astype('int')
        self.n_groups = self.n_classes * self.n_places
        self.group_counts = (
                torch.arange(self.n_groups).unsqueeze(1) == torch.from_numpy(self.group_array)).sum(1).float()
        self.y_counts = (
                torch.arange(self.n_classes).unsqueeze(1) == torch.from_numpy(self.y_array)).sum(1).float()
        self.p_counts = (
                torch.arange(self.n_places).unsqueeze(1) == torch.from_numpy(self.p_array)).sum(1).float()
        self.filename_array = self.metadata_df['isic_id'].values

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]
        p = self.confounder_array[idx]
        n = self.filename_array[idx]

        img_path = os.path.join(self.basedir, self.filename_array[idx])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, y, g, p, n
    

def get_transform(target_resolution, train, augment_data):
    scale = 256.0 / 224.0

    if (not train) or (not augment_data):
        # Resizes the image to a slightly larger square then crops the center.
        transform = transforms.Compose([
            transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333)),
                #interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform


def get_loader(data, train, **kwargs):
    if not train: # Validation or testing
        shuffle = False
        sampler = None
    else: # Training
        shuffle = True
        sampler = None
    print('data len: ',len(data))
    loader = DataLoader(
        data,
        shuffle=shuffle,
        sampler=sampler,
        **kwargs)
    return loader

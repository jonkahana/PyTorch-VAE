import os
from os.path import join

import PIL.Image
import numpy as np
import torch
import torchvision.transforms
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
import zipfile


def load_np_data(filepath, data_name):
    data = dict(np.load(filepath, allow_pickle=True))

    if 'n_classes' in data.keys() and not len(data['n_classes'].shape) > 0:
        data['n_classes'] = int(data['n_classes'])
    imgs = data['imgs'].astype(np.float32)
    if 'cars3d' not in data_name:
        imgs = imgs / 255.0
    data['imgs'] = imgs
    return data


# Add your custom dataset class here
class MyDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.
    
    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """

    def _check_integrity(self) -> bool:
        return True


class Numpy_Dataset(Dataset):

    def __init__(self, data_name, transform=None, is_train=True):

        # self.data_folder = '/hdd/projects/disentanglement/lord/preprocess'
        self.data_folder = '/cs/labs/yedid/jonkahana/experiments/lord/preprocess'
        if is_train:
            self.data_name = data_name
        else:
            self.data_name = data_name.replace('train', 'test')
        self.data = load_np_data(join(self.data_folder, self.data_name + '.npz'), self.data_name)
        self.images = self.data['imgs']
        self.target = self.data['classes']
        self.transform = transform
        if self.transform is None:
            self.transform = torchvision.transforms.ToTensor()

    def __getitem__(self, idx):
        np_image = self.images[idx]
        single_channel = False
        if np_image.shape[-1] == 1:
            single_channel = True
            np_image = np.concatenate([np_image] * 3, axis=-1)
        pil_img = PIL.Image.fromarray(np_image)
        img = self.transform(pil_img)
        if single_channel:
            img = torch.mean(img, dim=0)
        return img, torch.tensor(self.target[idx])

    def __len__(self):
        return len(self.target)


class OxfordPets(Dataset):
    """
    URL = https://www.robots.ox.ac.uk/~vgg/data/pets/
    """

    def __init__(self,
                 data_path: str,
                 split: str,
                 transform: Callable,
                 **kwargs):
        self.data_dir = Path(data_path) / "OxfordPets"
        self.transforms = transform
        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.jpg'])

        self.imgs = imgs[:int(len(imgs) * 0.75)] if split == "train" else imgs[int(len(imgs) * 0.75):]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])

        if self.transforms is not None:
            img = self.transforms(img)

        return img, 0.0  # dummy datat to prevent breaking


class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
            self,
            data_path: str,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            patch_size: Union[int, Sequence[int]] = (256, 256),
            num_workers: int = 0,
            pin_memory: bool = False,
            **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None, data_name='numpy') -> None:

        if data_name == 'oxford':
            train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                   transforms.CenterCrop(self.patch_size),
                                                   #                                               transforms.Resize(self.patch_size),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

            val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                 transforms.CenterCrop(self.patch_size),
                                                 #                                             transforms.Resize(self.patch_size),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

            self.train_dataset = OxfordPets(
                self.data_dir,
                split='train',
                transform=train_transforms,
            )

            self.val_dataset = OxfordPets(
                self.data_dir,
                split='val',
                transform=val_transforms,
            )
        elif data_name == 'celeba':
            train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                   transforms.CenterCrop(148),
                                                   transforms.Resize(self.patch_size),
                                                   transforms.ToTensor(), ])

            val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                 transforms.CenterCrop(148),
                                                 transforms.Resize(self.patch_size),
                                                 transforms.ToTensor(), ])

            self.train_dataset = MyCelebA(
                self.data_dir,
                split='train',
                transform=train_transforms,
                download=False,
            )

            # Replace CelebA with your dataset
            self.val_dataset = MyCelebA(
                self.data_dir,
                split='test',
                transform=val_transforms,
                download=False,
            )
        else:
            train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor()])

            val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor()])

            self.train_dataset = Numpy_Dataset(data_name, transform=train_transforms, is_train=True)
            self.val_dataset = Numpy_Dataset(data_name, transform=val_transforms, is_train=False)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )


if __name__ == '__main__':
    data = Numpy_Dataset('cars3d_train')

    a = 1
from __future__ import print_function
import torch.utils.data as data
from torchvision import datasets
from PIL import Image
import numpy as np

class Dataset(data.Dataset):
    """Args:
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    def __init__(self, dataset, root_folder, transform=None):
        self.root_folder = root_folder

        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,download=True),

                          'mnist': lambda: datasets.MNIST(self.root_folder, train=True,download=True),

                          'usps': lambda: datasets.USPS(self.root_folder, train=True,download=True),

                          'svhn': lambda: datasets.SVHN(self.root_folder, split='train',download=True),

                          'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',download=True)}

        self.transform = transform
        self.dataset = valid_datasets[dataset]()

    def __getitem__(self, index):
        """
         Args:
             index (int): Index
         Returns:
             tuple: (image, target) where target is index of the target class.
         """

        # img, target = self.data[index], self.labels[index]
        img, target = self.dataset[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # print(img.size)
        # print(type(img), target)
        img = np.asarray(img)

        # if img.shape[2] != 1:
        if len(img.shape) == 3:
        # if img.size[0] != 1:
            #print(img)
            # img = Image.fromarray(np.uint8(np.asarray(img.transpose((1, 2, 0)))))

            img = Image.fromarray(np.uint8(img))
        #
        # elif img.shape[2] == 1:
        elif len(img.shape) == 2:
            # im = np.uint8(np.asarray(img))

            im = np.uint8(img)
            # print(np.vstack([im,im,im]).shape)
            # im = np.vstack([im, im, im]).transpose((1, 2, 0))
            im = np.vstack([im, im, im])
            img = Image.fromarray(im)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        if self.transform is not None:
            img = self.transform(img)
            #  return img, target

        print(img.size)
        return img, target
        
    def __len__(self):
        return len(self.dataset)

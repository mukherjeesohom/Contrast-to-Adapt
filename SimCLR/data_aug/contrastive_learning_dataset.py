from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
from data_reader import Dataset


class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, name, n_views):
        # valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
        #                                                       transform=ContrastiveLearningViewGenerator(
        #                                                           self.get_simclr_pipeline_transform(32),
        #                                                           n_views),
        #                                                       download=True),

        #                   'mnist': lambda: datasets.MNIST(self.root_folder, train=True,
        #                                                   transform=ContrastiveLearningViewGenerator(
        #                                                       self.get_simclr_pipeline_transform(28),
        #                                                       n_views),
        #                                                   download=True),

        #                   'usps': lambda: datasets.USPS(self.root_folder, train=True,
        #                                                   transform=ContrastiveLearningViewGenerator(
        #                                                       self.get_simclr_pipeline_transform(16),
        #                                                       n_views),
        #                                                   download=True),

        #                   'svhn': lambda: datasets.SVHN(self.root_folder, split='train',
        #                                                   transform=ContrastiveLearningViewGenerator(
        #                                                       self.get_simclr_pipeline_transform(32),
        #                                                       n_views),
        #                                                   download=True),

        #                   'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
        #                                                   transform=ContrastiveLearningViewGenerator(
        #                                                       self.get_simclr_pipeline_transform(96),
        #                                                       n_views),
        #                                                   download=True)}

        valid_transf = {'cifar10': lambda: ContrastiveLearningViewGenerator(self.get_simclr_pipeline_transform(32),n_views),
                          'mnist': lambda: ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(28),
                                                              n_views),
                          'usps': lambda: ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(16),
                                                              n_views),
                          'svhn': lambda:ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(32),
                                                              n_views),
                          'stl10': lambda:ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(96),
                                                              n_views),
                                                          }



        try:
            transf_fn = valid_transf[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return Dataset(name, self.root_folder, transform=transf_fn())

import json
import os
import pickle
import random
from scipy.io import loadmat
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader


def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


def save_preds(exp, probability, clean):
    name = './stats/cifar100/stats{}.pcl'
    nm = name.format(exp)
    if os.path.exists(nm):
        probs_history, clean_history = pickle.load(open(nm, "rb"))
    else:
        probs_history, clean_history = [], []
    probs_history.append(probability)
    clean_history.append(clean)
    pickle.dump((probs_history, clean_history), open(nm, "wb"))



class cifar_dataset(Dataset):
    def __init__(self, dataset, r, noise_mode, root_dir, transform, mode, noise_file='', pred=[], probability=[],
                 log='', oracle='none', mix_labelled=True):
        assert oracle in ('none', 'positive', 'negative', 'all', 'negative_shuffle')
        assert dataset in ('cifar10', 'mnist', 'usps')
        without_class = False

        self.r = r  # noise ratio
        self.transform = transform
        self.mode = mode


        self.mix_labelled = mix_labelled
        self.num_classes = 10 if dataset == 'cifar10' else 100

	# loading test images and labels
        if self.mode == 'test':

            if dataset == 'cifar10':
                test_dic = unpickle('%s/test_batch' % root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['labels']

            elif dataset == 'mnist':

                mnist_data = loadmat('./datasets/mnist_data.mat')
                mnist_test = np.reshape(mnist_data['test_32'], (10000, 32, 32, 1))
                mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)
                test_data = mnist_test.transpose(0, 1, 2, 3).astype(np.float32)

                mnist_labels_test = mnist_data['label_test']
                test_label = list(np.argmax(mnist_labels_test, axis=1))
                test_label = test_label

        else:
            train_data = []
            train_label = []
            if dataset == 'cifar10':
                for n in range(1, 6):
                    dpath = '%s/data_batch_%d' % (root_dir, n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label + data_dic['labels']

                train_data = np.concatenate(train_data)

                print(f'cifar-10 org_data shape: {train_data.shape}')

                train_data = train_data.reshape((50000, 3, 32, 32))
                train_data = train_data.transpose((0, 2, 3, 1))

                print(f'cifar-10 final data: shape {train_data.shape}')

            elif dataset == 'mnist':

                mnist_data = loadmat('./datasets/mnist_data.mat')

                mnist_train = np.reshape(mnist_data['train_32'], (55000, 32, 32, 1))
                mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
                mnist_train = mnist_train.transpose(0, 1, 2, 3).astype(np.float32)
                mnist_labels_train = mnist_data['label_train']

                train_label = np.argmax(mnist_labels_train, axis=1)
                inds = np.random.permutation(mnist_train.shape[0])

                train_data = mnist_train[inds]
                train_label = list(train_label[inds])

                print(f'mnist org_data shape: {train_data.shape}')
                train_data = (train_data*255).astype(np.uint8)


                print(f'mnist final data shape: {train_data.shape}')

            # Loading noisy labels [size of the list = training set]

            if os.path.exists(noise_file):
                noise_label = json.load(open(noise_file, "r"))


            self.clean = (np.array(noise_label) == np.array(train_label))
            if self.mode == 'all':
                self.train_data = train_data
                self.noise_label = noise_label
                self.train_label = train_label
            else:
                clean = (np.array(noise_label) == np.array(train_label))

                if oracle == 'negative':
                    pred = pred * (clean == 1)  # don't take noisy
                elif oracle == 'negative_shuffle':
                    pred_clean = (pred == 1) * (clean == 0)  # shuffle labels of FP
                    noise_label = np.array(noise_label)
                    noise_label[pred_clean] = np.random.randint(0, self.num_classes, len(noise_label[pred_clean]))
                elif oracle == 'positive':
                    pred = (pred + clean) > 0  # take all clean
                elif oracle == 'all':
                    pred = clean  # take only clean

                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]

                    auc = roc_auc_score(clean, probability) if self.r > 0 else 1
                    tp, fp, fn = (np.equal(pred, clean) * (clean == 1)).sum(), \
                                 (np.not_equal(pred, clean) * (clean == 0)).sum(), \
                                 (np.not_equal(pred, clean) * (clean == 1)).sum()
                    # pc,nc = (clean==1).sum(), (clean==0).sum()
                    log.write('Number of labeled samples:%d\t'
                              'AUC:%.3f\tTP:%.3f\tFP:%.3f\tFN:%.3f\t'
                              'Noise in labeled dataset:%.3f\n' % (
                                  pred.sum(), auc, tp, fp, fn, fp / (tp + fp)))

                    log.flush()

                elif self.mode == "unlabeled":
                    pred_idx = (1 - pred).nonzero()[0]

                self.train_data = train_data[pred_idx]
                self.noise_label = [noise_label[i] for i in pred_idx]
                print("%s data has a size of %d" % (self.mode, len(self.noise_label)))

    def __getitem__(self, index):
        if self.mode == 'labeled':
            img, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2, target, index, prob if self.mix_labelled else target
        elif self.mode == 'unlabeled':
            img = self.train_data[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2
        elif self.mode == 'all':
            img, target, clean = self.train_data[index], self.noise_label[index], self.train_label[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2, target, index, clean
        elif self.mode == 'test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target
        elif self.mode == 'perf_on_train':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target

    def __len__(self):
        if self.mode != 'test':
            return len(self.train_data)
        else:
            return len(self.test_data)


class cifar_dataloader():
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir, log, noise_file='',
                 stronger_aug=False):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file
        if self.dataset == 'cifar10':
            self.transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            self.transform_warmup = transforms.Compose([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=15.,
                                        translate=(0.1, 0.1),
                                        scale=(2. / 3, 3. / 2),
                                        shear=(-0.1, 0.1, -0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        elif self.dataset == 'mnist':
            self.transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            self.transform_warmup = transforms.Compose([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=15.,
                                        translate=(0.1, 0.1),
                                        scale=(2. / 3, 3. / 2),
                                        shear=(-0.1, 0.1, -0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        self.transform_warmup = self.transform_warmup if stronger_aug else self.transform_train
        self.clean = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                   root_dir=self.root_dir, transform=self.transform_warmup, mode="all",
                                   noise_file=self.noise_file).clean

    def run(self, mode, pred=[], prob=[]):
        if mode == 'warmup':
            all_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                        root_dir=self.root_dir, transform=self.transform_warmup, mode="all",
                                        noise_file=self.noise_file)
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size * 2,
                shuffle=True,
                num_workers=self.num_workers)
            return trainloader

        elif mode == 'train':
            labeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                            root_dir=self.root_dir, transform=self.transform_train, mode="labeled",
                                            noise_file=self.noise_file, pred=pred, probability=prob, log=self.log)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)

            unlabeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                              root_dir=self.root_dir, transform=self.transform_train, mode="unlabeled",
                                              noise_file=self.noise_file, pred=pred)
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            return labeled_trainloader, unlabeled_trainloader

        # Putting newly created mode i.e. perf_on_train since we want to see the performance on the train set.

        elif mode == 'test':
            test_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                         root_dir=self.root_dir, transform=self.transform_test, mode='perf_on_train')
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return test_loader

        elif mode == 'eval_train':
            eval_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                         root_dir=self.root_dir, transform=self.transform_test, mode='all',
                                         noise_file=self.noise_file)
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return eval_loader

import sys

sys.path.append('../loader')
# from unaligned_data_loader import UnalignedDataLoader
from datasets.svhn import load_svhn
from datasets.mnist import load_mnist
from datasets.usps import load_usps
# from gtsrb import load_gtsrb
# from synth_traffic import load_syntraffic

from datasets.create_dataloader import create_DataLoader


def return_dataset(data, scale=False, usps=False, all_use='no'):
    if data == 'svhn':
        train_image, train_label, \
        test_image, test_label = load_svhn()
    if data == 'mnist':
        train_image, train_label, \
        test_image, test_label = load_mnist(scale=scale, usps=usps, all_use=all_use)
        print(train_image.shape)
    if data == 'usps':
        train_image, train_label, \
        test_image, test_label = load_usps(all_use=all_use)
    # if data == 'synth':
    #     train_image, train_label, \
    #     test_image, test_label = load_syntraffic()
    # if data == 'gtsrb':
    #     train_image, train_label, \
    #     test_image, test_label = load_gtsrb()

    return train_image, train_label, test_image, test_label

# we don't need target just source
def dataset_read(source, target, batch_size, scale=False, all_use='no'):
    # Return train and test loader

    S = {}
    S_test = {}
    # T = {}
    # T_test = {}
    usps = False
    if source == 'usps': # or target == 'usps':
        usps = True

    train_source, s_label_train, test_source, s_label_test = return_dataset(source, scale=scale,
                                                                            usps=usps, all_use=all_use)
    # train_target, t_label_train, test_target, t_label_test = return_dataset(target, scale=scale, usps=usps,
                                                                            # all_use=all_use)


    S['imgs'] = train_source
    S['labels'] = s_label_train
    # T['imgs'] = train_target
    # T['labels'] = t_label_train

    # input target samples for both 
    S_test['imgs'] = test_source
    S_test['labels'] = s_label_test
    # T_test['imgs'] = test_target
    # T_test['labels'] = t_label_test
    scale = 40 if source == 'synth' else 28 if source == 'usps' or target == 'usps' else 32
    # scale = 40 if source == 'synth' else 28 if source == 'usps' else 32
    # train_loader = UnalignedDataLoader()

    train_loader = create_DataLoader(S, batch_size, scale=scale, shuffle=False, )
    # dataset = train_loader.load_data()
    # test_loader = UnalignedDataLoader()
    
    val_loader = create_DataLoader(S_test, batch_size, scale=scale, shuffle=False)
    # dataset_test = test_loader.load_data()
    return train_loader, val_loader

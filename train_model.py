import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import matplotlib.pyplot as plt

from datasets.dataset_read import dataset_read
from model.build_gen import CustLeNet

def train_val_loop(model, epochs:int, train_loader, val_loader, optimizer, criterion, bs:int, device, chk_dir, save_epoch, print_interval, dataset, scheduler=None):
    """
        This function will perform train loop (forward-backward pass) and also evaluate performance 
        on validation data after each epoch of training. Finally losses will be printed out.
    Return:
        It returns two list containing training and validation loss
    Args:
        model: neural network model to be train
        epochs: number of epochs(times) train the model over complete train data set
        train_loader: data loader for train set
        val_loader: data loader for validation set
        optimizer: optimizer to update model parameters
        criterion: loss function to evaluate the training through loss
        bs: batch size (number of images grouped in a batch)
        device: device to which tensors will be allocated (in our case, from gpu 0 to 7)
        scheduler: update the learning rate based on chosen scheme if provided
    """
    print("Training Started !!")

    # store the losses after every epoch 
    loss_train = []
    loss_val = []
    
    for epoch in range(epochs):
        #Training
        model.train()
        running_loss = 0

        for batch_idx, samples in enumerate(train_loader):
            inputs = samples[0].to(device)
            labels = samples[1].to(device).long()
            # labels = labels.squeeze(1
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            ###accumulating loss for each batch
            running_loss += loss.item()
            
            if scheduler:
                # changing LR
                scheduler.step()

            if batch_idx % print_interval == 0: # intermediate progress printing
                print("Epoch{}, iter{}, running loss: {}".format(epoch, batch_idx, running_loss/(bs*(batch_idx+1))))

        loss_train.append(running_loss/len(train_loader))

        print("Epoch{}, Training loss: {}".format(epoch, running_loss/len(train_loader)))
        
        if epoch % save_epoch == 0:
            torch.save(model.state_dict(), f'{chk_dir}/{dataset}_epoch_{epoch}.pth')

        #Validation
        model.eval()
        running_loss_val = 0
        for i, samples in enumerate(val_loader):
            inputs = samples[0].to(device)
            labels = samples[1].to(device).long()
            # labels = labels.squeeze(1)

            with torch.no_grad(): 
                outputs = model(inputs)
                # loss = criterion(outputs,labels.long())
                loss = criterion(outputs,labels)

                ###accumulating loss for each batch
                running_loss_val += loss.item()

            #if i%10 == 0:
        loss_val.append(running_loss_val/len(val_loader))
        print("epoch{}, Validation loss: {}".format(epoch, running_loss_val/len(val_loader)))
        
    return loss_train, loss_val


if __name__ == '__main__':

    # Training settings
    parser = argparse.ArgumentParser(description='Pretraining')
    parser.add_argument('--all_use', type=str, default='no', metavar='N',
                        help='use all training data? in usps adaptation')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', metavar='N',
                        help='source only or not')
    # parser.add_argument('--eval_only', action='store_true', default=False,
                        # help='evaluation only option')
    parser.add_argument('--lr', type=float, default=0.0002, metavar='LR',
                        help='learning rate (default: 0.0002)')
    parser.add_argument('--max_epoch', type=int, default=200, metavar='N',
                        help='how many epochs')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    # parser.add_argument('--num_k', type=int, default=4, metavar='N',
                        # help='hyper paremeter for generator update')
    # parser.add_argument('--one_step', action='store_true', default=False,
                        # help='one step training with gradient reversal layer')
    # parser.add_argument('--optimizer', type=str, default='adam', metavar='N', help='which optimizer')
    # parser.add_argument('--resume_epoch', type=int, default=100, metavar='N',
                        # help='epoch to resume')
    parser.add_argument('--save_epoch', type=int, default=10, metavar='N',
                        help='when to restore the model')
    # parser.add_argument('--save_model', action='store_true', default=False,
    #                     help='save_model or not')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--source', type=str, default='svhn', metavar='N',
                        help='source dataset')
    parser.add_argument('--target', type=str, default='mnist', metavar='N', help='target dataset')
    # parser.add_argument('--use_abs_diff', action='store_true', default=False,
                        # help='use absolute difference value as a measurement')
 
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
 
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
 
    print(args)

    batch_size = args.batch_size
    source = args.source
    # source = 'mnist'
    target = args.target
    # num_k = args.num_k
    checkpoint_dir = args.checkpoint_dir
    save_epoch = args.save_epoch
    # use_abs_diff = args.use_abs_diff
    all_use = args.all_use
    print_interval = 100

    if source == 'svhn':
        scale = True
    else:
        scale = False

    # optimizer = args.optimizer
    # LR = args.lr
    
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)

    print('dataset loading')
    train_loader, val_loader = dataset_read(source, target, batch_size, scale, all_use)
    print('load finished!')

    device = torch.device("cuda")

    model = CustLeNet(source, target)

    optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay=0.0001)

    criterion = nn.CrossEntropyLoss().cuda()

    torch.cuda.manual_seed(1) # fixing seed according to MCD work

    loss_train, loss_val = train_val_loop(model.cuda(), args.max_epoch, train_loader, val_loader,
                    optimizer, criterion, batch_size, device,
                    checkpoint_dir, save_epoch, print_interval, source+target)

    # create and save the plot
    x = range(1, args.max_epoch+1)
    plt.title("Plot showing training and validation loss against number of epochs")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.plot(x, loss_train, color='b', label='Training loss')
    plt.plot(x, loss_val, color='r', label='Validation loss')
    plt.legend()
    plt.savefig(f'checkpoint/{source+target}_loss_curve.png', bbox_inches='tight')
    plt.show()

import model.svhn2mnist
import model.usps
# import model.syn2gtrsb
# import model.syndig2svhn

import torch.nn as nn

def Generator(source, target, pixelda=False):
    if source == 'usps' or target == 'usps':
        return model.usps.Feature()
    elif source == 'svhn':
        return model.svhn2mnist.Feature()
    # elif source == 'synth':
    #     return syn2gtrsb.Feature()


def Classifier(source, target):
    if source == 'usps' or target == 'usps':
        return model.usps.Predictor()
    if source == 'svhn':
        return model.svhn2mnist.Predictor()
    # if source == 'synth':
    #     return syn2gtrsb.Predictor()

class CustLeNet(nn.Module):
    '''
    Create NN architecture based on the source and target datasets - basically modified version of LeNet.
    '''

    def __init__(self, source, target):
        super().__init__()
        self.feature = Generator(source, target)
        self.clf = Classifier(source, target)

    def forward(self, x):
        x = self.feature(x)
        x = self.clf(x)
        return x
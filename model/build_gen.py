import svhn2mnist
import usps
import syn2gtrsb
import syndig2svhn

import torch.nn as nn

def Generator(source, target, pixelda=False):
    if source == 'usps' or target == 'usps':
        return usps.Feature()
    elif source == 'svhn':
        return svhn2mnist.Feature()
    # elif source == 'synth':
    #     return syn2gtrsb.Feature()


def Classifier(source, target):
    if source == 'usps' or target == 'usps':
        return usps.Predictor()
    if source == 'svhn':
        return svhn2mnist.Predictor()
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
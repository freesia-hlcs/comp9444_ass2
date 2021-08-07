#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables train_val_split,
batch_size as well as the transform function.
You are encouraged to modify these to improve the performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


"""
   Answer to Question:

Briefly describe how your program works, and explain any design and training
decisions you made along the way.

"""

############################################################################
######     Specify transform(s) to be applied to the input images     ######
############################################################################
def transform(mode):
    """
    Called when loading the data. Visit this URL for more information:
    https://pytorch.org/vision/stable/transforms.html
    You may specify different transforms for training and testing
    """
    if mode == 'train':
        transform = transforms.Compose([transforms.Resize(64),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))],
                                       )
        return transform
    elif mode == 'test':
        transform = transforms.Compose([transforms.Resize(64),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))],
                                       )
        return transform

############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )

        self.linear_layers = nn.Sequential(
            nn.Linear(16*4*4, 14)
        )
        '''

        # torch.Size([256, 3, 224, 224])
        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 16, kernel_size=5, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 16, kernel_size=5, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),


        )

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)              # 34


        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

        )

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)              # 19

        self.cnn_layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),


        )

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)            # 12


        self.net = nn.Sequential(self.cnn_layer1, self.pool1, self.cnn_layer2, self.pool2, self.cnn_layer3, self.pool3)


        self.linear_layers = nn.Sequential(
            nn.Linear(32*64, 14*32),
            nn.ReLU(),
            nn.Linear(14*32, 14)

        )
        '''
        
    def forward(self, t):
        x = self.cnn_layers(t)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)

        return x
        '''

        output = self.net(t)
        print(t.size())
        output = output.view(output.size(0), -1)
        print(output.size())
        output = self.linear_layers(output)

        return output

        '''

class loss(nn.Module):
    """
    Class for creating a custom loss function, if desired.
    If you instead specify a standard loss function,
    you can remove or comment out this class.
    """
    def __init__(self):
        super(loss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, output, target):
        result = self.loss(output, target)
        return result


net = Network()
lossFunc = loss()
############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data"
train_val_split = 0.9
batch_size = 64
epochs = 50
optimiser = optim.Adam(net.parameters(), lr=0.001)




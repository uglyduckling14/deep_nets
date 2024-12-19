##################################
# module: lenet.py
##################################
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten

'''
Here's what we're importing.
Conv2d --  PyTorch’s implementation of convolutional layers
Linear --  Fully connected layers
MaxPool2d --  2D max-pooling layer
ReLU -- ReLU activation function
LogSoftmax -- returns the predicted probabilities of each class 
flatten: Flattens the output of a multi-dimensional volume
so that a fully connected layer can be connected to it.
'''

class LeNet(Module):
    def __init__(self, numChannels, classes):
        # call the parent constructor
        super(LeNet, self).__init__()

        # initialize first set of CONV => RELU => POOL layers
        '''
        Initialize our first set of CONV => RELU => POOL layers. The first 
        CONV layer learns a total of 20 filters, each of which are 5x5. A 
        ReLU activation function is then applied, followed by a 2x2 max-pooling 
        layer with a 2x2 stride to reduce the spatial dimensions of our input image.
        '''
        self.conv1 = Conv2d(in_channels=numChannels, out_channels=20, kernel_size=(5, 5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # initialize second set of CONV => RELU => POOL layers
        '''
        There's a second set of CONV => RELU => POOL layers on Lines 22-25. 
        The number of filters is increased, but 
        the 5x5 kernel size is maintained. ReLU activation is applied, followed by max-pooling.
        '''
        self.conv2 = Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # initialize first (and only) set of FC => RELU layers
        '''
        Next comes the set of fully connected layers.
        We define the number of inputs to the layer (800) along with the desired number 
        of output nodes (500). A ReLu activation follows the FC layer.
        '''
        self.fc1 = Linear(in_features=800, out_features=500)
        self.relu3 = ReLU()

        # initialize our softmax classifier
        '''
        Finally, we apply the softmax classifier. The number of in_features
        is set to 500, which is the output dimensionality from the previous layer. We then 
        apply LogSoftmax to get the predicted probabilities of each class.
        '''
        self.fc2 = Linear(in_features=500, out_features=classes)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        # pass the input through the first CONV => RELU => POOL
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        # pass the output through the second CONV => RELU => POOL
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        # flatten the output from the previous layer and pass it
        # through FC => RELU
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        # pass the output to softmax classifier and obtain output
        # predictions
        x = self.fc2(x)
        output = self.logSoftmax(x)
        return output

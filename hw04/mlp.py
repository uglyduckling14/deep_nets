######################################
# mlp.py
# a few examples of building ANNs
# for CS5600/6600: F24: HW04 Coding Lab
#
# bugs to vladimir kulyukin on canvas
#######################################

from collections import OrderedDict
import torch.nn as nn

## hiddenActFun1=nn.Hardtanh(), nn.Sigmoid(), nn.ReLU()
## this builds 4 x 8 x 3 ANN with ReLU activation function in the hidden
## layer of 8 neurons.
def build_4_8_3_mlp_model(inFeatures=4, hiddenDim1=8, hiddenActFun1=nn.ReLU(), nbClasses=3):
    mlp_model = nn.Sequential(OrderedDict([
	("hidden_layer_1", nn.Linear(inFeatures, hiddenDim1)),
	("activation_1",   hiddenActFun1),
	("output_layer",   nn.Linear(hiddenDim1, nbClasses))
    ]))
    # return the sequential model
    return mlp_model

## this builds 4 x 8 x 8 x 3 with ReLU activaion on the first hidden layer of 8 neurons.
## and Sigmoid on the second hidden layer of 8 neurons.
def build_4_8_8_3_mlp_model(inFeatures=4, hiddenDim1=8, hiddenDim2=8, hiddenActFun1=nn.ReLU(),
                            hiddenActFun2=nn.Sigmoid(), nbClasses=3):
    mlp_model = nn.Sequential(OrderedDict([
	("hidden_layer_1", nn.Linear(inFeatures, hiddenDim1)),
	("activation_1",   hiddenActFun1),
        ("hidden_layer_2", nn.Linear(hiddenDim1, hiddenDim2)),
        ("activation_2",   hiddenActFun2),        
	("output_layer",   nn.Linear(hiddenDim2, nbClasses))
    ]))
    # return the sequential model
    return mlp_model




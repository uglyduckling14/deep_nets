######################################
# mlp_hw.py
# CS5600/6600: F24: HW04: Some
# starter code for Problem 2
#
# bugs to vladimir kulyukin in canvas
#######################################

from collections import OrderedDict
import torch.nn as nn

### =============== 10x20x5 with ReLU and Sigmoid =======================

# 1) initialize 10x20x5 model with ReLU activation function in the hidden layer.
def build_10_20_5_mlp_relu_model(inFeatures=10, hiddenDim1=20, hiddenActFun1=nn.ReLU(),
                                 nbClasses=5):
    model = nn.Sequential(
        nn.Linear(inFeatures, hiddenDim1),
        hiddenActFun1,
        nn.Linear(hiddenDim1, nbClasses)
    )
    return model

# 2) initialize 10x20x5 model with Sigmoid activation function in the hidden layer.
def build_10_20_5_mlp_sigmoid_model(inFeatures=10, hiddenDim1=20, hiddenActFun1=nn.Sigmoid(),
                                 nbClasses=5):
    model = nn.Sequential(
        nn.Linear(inFeatures, hiddenDim1),
        hiddenActFun1,
        nn.Linear(hiddenDim1, nbClasses)
    )
    return model

### ================ 10x20x20x5 with ReLU and Sigmoid ======================

# 3) initialize 10x20x20x5 model with ReLU activation function in the hidden layer.
def build_10_20_20_5_mlp_relu_model(inFeatures=10, hiddenDim1=20, hiddenDim2=20,
                                    hiddenActFun1=nn.ReLU(),
                                    hiddenActFun2=nn.ReLU(),
                                    nbClasses=5):
    model = nn.Sequential(
        nn.Linear(inFeatures, hiddenDim1),
        hiddenActFun1,
        nn.Linear(hiddenDim1, hiddenDim2),
        hiddenActFun2,
        nn.Linear(hiddenDim2, nbClasses)
    )
    return model

# 4) initialize 10x20x20x5 model with Sigmoid activation function in the hidden layer.
def build_10_20_20_5_mlp_sigmoid_model(inFeatures=10, hiddenDim1=20, hiddenDim2=20,
                                    hiddenActFun1=nn.Sigmoid(),
                                    hiddenActFun2=nn.Sigmoid(),
                                    nbClasses=5):
    model = nn.Sequential(
        nn.Linear(inFeatures, hiddenDim1),
        hiddenActFun1,
        nn.Linear(hiddenDim1, hiddenDim2),
        hiddenActFun2,
        nn.Linear(hiddenDim2, nbClasses)
    )
    return model

### ================= 10x20x20x20x5 with ReLU and Sigmoid ====================

# 5) initialize 10x20x20x20x5 model with ReLU activation function in the hidden layer.
def build_10_20_20_20_5_mlp_relu_model(inFeatures=10,
                                       hiddenDim1=20,
                                       hiddenDim2=20,
                                       hiddenDim3=20,
                                       hiddenActFun1=nn.ReLU(),
                                       hiddenActFun2=nn.ReLU(),
                                       hiddenActFun3=nn.ReLU(),
                                       nbClasses=5):
    model = nn.Sequential(
        nn.Linear(inFeatures, hiddenDim1),
        hiddenActFun1,
        nn.Linear(hiddenDim1, hiddenDim2),
        hiddenActFun2,
        nn.Linear(hiddenDim2, hiddenDim3),
        hiddenActFun3,
        nn.Linear(hiddenDim3, nbClasses)
    )
    return model

# 6) initialize 10x20x20x20x5 model with Sigmoid activation function in the hidden layer.
def build_10_20_20_20_5_mlp_sigmoid_model(inFeatures=10,
                                          hiddenDim1=20,
                                          hiddenDim2=20,
                                          hiddenDim3=20,
                                          hiddenActFun1=nn.Sigmoid(),
                                          hiddenActFun2=nn.Sigmoid(),
                                          hiddenActFun3=nn.Sigmoid(),
                                          nbClasses=5):
    model = nn.Sequential(
        nn.Linear(inFeatures, hiddenDim1),
        hiddenActFun1,
        nn.Linear(hiddenDim1, hiddenDim2),
        hiddenActFun2,
        nn.Linear(hiddenDim2, hiddenDim3),
        hiddenActFun3,
        nn.Linear(hiddenDim3, nbClasses)
    )
    return model




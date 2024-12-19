###################################
## train_lenet.py
## training LeNet on KMNIST
###################################

import matplotlib
from lenet import LeNet
from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
from torch.optim import Adam
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time

matplotlib.use('Agg')

### python train.py --model output/model.pth --plot output/plot.png
ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', type=str, required=True, help='path to persisted trained model')
ap.add_argument('-p', '--plot', type=str, required=True, help='path to saved loss/accuracy plot')
args = vars(ap.parse_args())

# define training hyperparameters
INIT_LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 10
# define the train and val splits
TRAIN_SPLIT = 0.75
VAL_SPLIT = 1 - TRAIN_SPLIT
# set the device we will be using to train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load the KMNIST dataset
print('<DBG> loading KMNIST train and test datasets...')
trainData = KMNIST(root='data', train=True,  download=True, transform=ToTensor())
testData  = KMNIST(root='data', train=False, download=True, transform=ToTensor())

# calculate the train/validation split
print('<DBG> generating the train/validation split...')
numTrainSamples = int(len(trainData) * TRAIN_SPLIT)
numValSamples   = int(len(trainData) * VAL_SPLIT)
(trainData, valData) = random_split(trainData,
	                            [numTrainSamples, numValSamples],
	                            generator=torch.Generator().manual_seed(42))

# initialize the train, validation, and test data loaders
trainDataLoader = DataLoader(trainData, shuffle=True, batch_size=BATCH_SIZE)
valDataLoader   = DataLoader(valData, batch_size=BATCH_SIZE)
testDataLoader  = DataLoader(testData, batch_size=BATCH_SIZE)

# calculate steps per epoch for training and validation set
trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
valSteps   = len(valDataLoader.dataset) // BATCH_SIZE

# let's iniitialize LeNet
print('<DBG> initializing LeNet...')

'''
Since the KMNIST dataset is grayscale, numChannels=1. We 
can set the number of classes by calling dataset.classes
of our trainData. to(device) moves the model
to CPU or GPU.
'''

model = LeNet(numChannels=1, classes=len(trainData.dataset.classes)).to(device)

# initialize our optimizer and loss function
'''
initialize our optimizer and loss function. We'll use the 
Adam optimizer for training and the negative log-likelihood 
for our loss function.
'''
opt = Adam(model.parameters(), lr=INIT_LR)
lossFn = nn.NLLLoss()

# H is a dictionary to store training history
H = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

# we'll measure how long training is going to take
print('<DBG> training LeNet...')
startTime = time.time()

# main training loop.
for e in range(0, EPOCHS):
    # set the model in training mode
    model.train()
    # initialize the total training and validation loss
    total_train_loss = 0
    total_val_loss = 0
    # initialize the number of correct predictions in the training
    # and validation step
    train_correct = 0
    val_correct = 0
    # loop over the training set
    for (x, y) in trainDataLoader:
        # send the input to the device
        (x, y) = (x.to(device), y.to(device))
        # perform a forward pass and calculate the training loss
        pred = model(x)
        loss = lossFn(pred, y)
        # zero out the gradients, perform the backpropagation step,
        # and update the weights
        opt.zero_grad()
        loss.backward()
        opt.step()
        # add the loss to the total training loss so far and
        # calculate the number of correct predictions
        total_train_loss += loss
        train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    # switch off autograd for evaluation        
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()
        # loop over the validation set
        for (x, y) in valDataLoader:
            # send the input to the device
            (x, y) = (x.to(device), y.to(device))
            # make the predictions and calculate the validation loss
            pred = model(x)
            total_val_loss += lossFn(pred, y)
            # calculate the number of correct predictions
            val_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    # Comput the stats:
    # calculate the average training and validation loss
    avg_train_loss = total_train_loss / trainSteps
    avg_val_loss = total_val_loss / valSteps
    # calculate the training and validation accuracy
    train_correct = train_correct / len(trainDataLoader.dataset)
    val_correct = val_correct / len(valDataLoader.dataset)
    
    # update training history
    H['train_loss'].append(avg_train_loss.cpu().detach().numpy())
    H['train_acc'].append(train_correct)
    H['val_loss'].append(avg_val_loss.cpu().detach().numpy())
    H['val_acc'].append(val_correct)
    
    # print the model training and validation information
    print('<DBG> EPOCH: {}/{}'.format(e + 1, EPOCHS))
    print('Train loss: {:.4f}, Train accuracy: {:.4f}'.format(avg_train_loss, train_correct))
    print('Val loss:   {:.4f}, Val accuracy: {:.4f}\n'.format(avg_val_loss, val_correct))


#finish measuring how long training took
endTime = time.time()
print('<DBG> total time taken to train the model: {:.2f}s'.format(endTime - startTime))

#we can now evaluate the network on the test set
print('<DBG> evaluating network...')

# turn off autograd for testing evaluation
with torch.no_grad():
    # set the model in evaluation mode
    model.eval()

    # initialize a list to store our predictions
    preds = []
    # loop over the test set
    for (x, y) in testDataLoader:
        # send the input to the device
        x = x.to(device)
        # make the predictions and add them to the list
        pred = model(x)
        preds.extend(pred.argmax(axis=1).cpu().numpy())

# let's generate a classification report
print(classification_report(testData.targets.cpu().numpy(),
	                    np.array(preds), target_names=testData.classes))

# let's plot the training loss/accuracy
plt.style.use('ggplot')
plt.figure()
plt.plot(H['train_loss'], label="train_loss")
plt.plot(H['val_loss'], label="val_loss")
plt.plot(H['train_acc'], label="train_acc")
plt.plot(H['val_acc'], label="val_acc")
plt.title('Training Loss and Accuracy on Dataset')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend(loc='lower left')
plt.savefig(args['plot'])

# we now persist the model to disk
torch.save(model, args['model'])

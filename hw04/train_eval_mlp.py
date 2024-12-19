############################################
## module: train_eval_mlp.py
## training and evaluating ANNs with PyTorch
## for Assignment 4 Coding Lab.
## 
## bugs to vladimir kulyukin in canvas.
############################################

import mlp
from torch.optim import SGD
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import torch.nn as nn
import torch

'''
The gen_next_batch()
function accepts three arguments:
    inputs:  Our input data to the neural network
    targets: Our target output values (i.e., what we want our neural network to accurately predict)
    batchSize: Size of data batch
'''
def gen_next_batch(inputs, targets, batchSize):
    
    for i in range(0, inputs.shape[0], batchSize):
        yield (inputs[i:i + batchSize], targets[i:i + batchSize])

# specify our batch size, number of epochs, and learning rate
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-2
# determine the device we will be using for training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("<DBG>: training using {}...".format(DEVICE))        

# generate a 3-class classification problem with 1000 data points,
# where each data point is a 4D feature vector
print("<DBG>: preparing data...")
(X, y) = make_blobs(n_samples=1000, n_features=4, centers=3,
                    cluster_std=2.5, random_state=13)
print('X = {}'.format(X[:5]))
print('y = {}'.format(y[:5]))

'''
X = [[ -2.17167054   9.62654054   3.41543784   1.64007163]
     [  1.810689     7.40108036   1.64578049   1.6343025 ]
     [ -7.60943821 -10.52535379   8.25352165   2.40327365]
     [ -2.30874824  10.61967835  -0.93221146   7.3008415 ]
     [ -6.30245516  -9.82620338   7.00373928   7.15240566]]
y = [1 1 0 2 0]
'''
      
# create training and testing splits, and convert them to PyTorch
# tensors
(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.20, random_state=13)
trainX = torch.from_numpy(trainX).float()
testX  = torch.from_numpy(testX).float()
trainY = torch.from_numpy(trainY).float()
testY  = torch.from_numpy(testY).float()
        
# initialize our model and display its architecture
mlp_4_8_3 = mlp.build_4_8_3_mlp_model().to(DEVICE)
print(mlp_4_8_3)
mlp_4_8_8_3 = mlp.build_4_8_8_3_mlp_model().to(DEVICE)
print(mlp_4_8_8_3)

'''
E.g., mpl_4_8_3 should print as follows:

Sequential(
  (hidden_layer_1): Linear(in_features=4, out_features=8, bias=True)
  (activation_1): ReLU()
  (output_layer): Linear(in_features=8, out_features=3, bias=True)
)
'''

# initialize optimizer and loss function
# opt = SGD(mlp_4_8_3.parameters(), lr=LR)
opt = SGD(mlp_4_8_8_3.parameters(), lr=LR)
lossFunc = nn.CrossEntropyLoss()

# create a template to summarize current training progress
trainTemplate = "epoch: {} test loss: {:.3f} test accuracy: {:.3f}"

print('------------ TRAINING ANN ----------------')

# now loop through the epochs
for epoch in range(0, EPOCHS):
    # initialize tracker variables and set our model to trainable
    print("<DBG>: epoch: {}...".format(epoch + 1))
    trainLoss = 0
    trainAcc = 0
    samples = 0
    #mlp_4_8_3.train()
    mlp_4_8_8_3.train()
    # loop over the current batch of data
    '''
    Move the batchX and batchY data to our CPU or GPU (depending on our DEVICE)
    Pass the batchX data through the neural and make predictions on it
    Use our loss function to compute our loss by comparing the output predictions
    to our ground-truth class labels
    '''
    for (batchX, batchY) in gen_next_batch(trainX, trainY, BATCH_SIZE):
	# flash data to the current device, run it through our
	# model, and calculate loss
        (batchX, batchY) = (batchX.to(DEVICE), batchY.to(DEVICE))
        #predictions = mlp_4_8_3(batchX)
        predictions = mlp_4_8_8_3(batchX)
        loss = lossFunc(predictions, batchY.long())
	# zero the gradients accumulated from the previous steps,
	# perform backpropagation, and update model parameters
        opt.zero_grad()
        loss.backward()
        opt.step()
	# update training loss, accuracy, and the number of samples
	# visited
        trainLoss += loss.item() * batchY.size(0)
        trainAcc += (predictions.max(1)[1] == batchY).sum().item()
        samples += batchY.size(0)
    # display model progress on the current training batch
    trainTemplate = "epoch: {} train loss: {:.3f} train accuracy: {:.3f}"
    print(trainTemplate.format(epoch + 1, (trainLoss / samples),
		               (trainAcc / samples)))

# ================ Evaluation =====================
# initialize tracker variables for testing, then set our model to
# evaluation mode
testLoss = 0
testAcc  = 0
samples  = 0
#mlp_4_8_3.eval()
mlp_4_8_8_3.eval()
# initialize a no-gradient context
print('------------ TESTING ANN ----------------')
with torch.no_grad():
    for epoch in range(0, EPOCHS):
        # loop over the current batch of test data
        for (batchX, batchY) in gen_next_batch(testX, testY, BATCH_SIZE):
            # flash the data to the current device
            (batchX, batchY) = (batchX.to(DEVICE), batchY.to(DEVICE))
            
            # run data through our model and calculate loss
            # predictions = mlp_4_8_3(batchX)
            predictions = mlp_4_8_8_3(batchX)
            loss = lossFunc(predictions, batchY.long())
            
            # update test loss, accuracy, and the number of
            # samples visited
            testLoss += loss.item() * batchY.size(0)
            testAcc  += (predictions.max(1)[1] == batchY).sum().item()
            samples  += batchY.size(0)
        
    # display model progress on the current test batch
    testTemplate = "epoch: {} test loss: {:.3f} test accuracy: {:.3f}"
    print(testTemplate.format(epoch + 1, (testLoss / samples),
			      (testAcc / samples)))
    print("")

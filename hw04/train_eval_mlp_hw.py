############################################
## training and evaluating ANNs with PyTorch.
## CS 5600/6600: F24: HW4: Problem 2
## 
## bugs to vladimir kulyukin in canvas.
############################################

import mlp_hw
from torch.optim import SGD
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import torch.nn as nn
import torch

def gen_next_batch(inputs, targets, batchSize):
    
    for i in range(0, inputs.shape[0], batchSize):
        yield (inputs[i:i + batchSize], targets[i:i + batchSize])

# specify our batch size, number of epochs, and learning rate
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-2 ## Learning Rate, i.e., eta.
# determine the device we will be using for training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("<DBG>: training using {}...".format(DEVICE))        

# generate a 3-class classification problem with 1000 data points,
# where each data point is a 4D feature vector
print("<DBG>: preparing data...")
### prepare 2000 samples with 10 features and 5 centers with the cluster STD = 1.75.
(X, y) = make_blobs(n_samples=2000, n_features=10, centers=5, cluster_std=1.75, random_state=13)
      
# do 70/30 train/test splits, and convert them to PyTorch sensors.
(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.30, random_state=13)
trainX = torch.from_numpy(trainX).float()
testX  = torch.from_numpy(testX).float()
trainY = torch.from_numpy(trainY).float()
testY  = torch.from_numpy(testY).float()
        
# initialize 10x20x5 ReLU model
ReLU_MLP1 = mlp_hw.build_10_20_5_mlp_relu_model().to(DEVICE)
# initialize 10x20x20x5 ReLU model
ReLU_MLP2 = mlp_hw.build_10_20_20_5_mlp_relu_model().to(DEVICE)
# initialize 10x20x20x20x5 ReLU model
ReLU_MLP3 = mlp_hw.build_10_20_20_20_5_mlp_relu_model().to(DEVICE)

# initialize 10x20x5 ReLU model
Sig_MLP1 = mlp_hw.build_10_20_5_mlp_sigmoid_model().to(DEVICE)
# initialize 10x20x20x5 ReLU model
Sig_MLP2 = mlp_hw.build_10_20_20_5_mlp_sigmoid_model().to(DEVICE)
# initialize 10x20x20x20x5 ReLU model
Sig_MLP3 = mlp_hw.build_10_20_20_20_5_mlp_sigmoid_model().to(DEVICE)

# initialize optimizers and loss function
ReLU_OPT1 = SGD(ReLU_MLP1.parameters(), lr=LR)
ReLU_OPT2 = SGD(ReLU_MLP2.parameters(), lr=LR)
ReLU_OPT3 = SGD(ReLU_MLP3.parameters(), lr=LR)

# initialize optimizers and loss function
Sig_OPT1 = SGD(Sig_MLP1.parameters(), lr=LR)
Sig_OPT2 = SGD(Sig_MLP2.parameters(), lr=LR)
Sig_OPT3 = SGD(Sig_MLP3.parameters(), lr=LR)

### let's use cross entry as the loss function for each model.
lossFunc = nn.CrossEntropyLoss()

# create a template to summarize current training progress
trainTemplate = "epoch: {} test loss: {:.3f} test accuracy: {:.3f}"

# now loop through the epochs
def train_mlp(model, opt, loss_func, num_epochs):
    model.train()
    print('------------ TRAINING ANN ----------------')
    for epoch in range(0, num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        # initialize stats variables and set our model to trainable
        print("<DBG>: epoch: {}...".format(epoch + 1))
        for (batchX, batchY) in gen_next_batch(trainX, trainY, BATCH_SIZE):
            # Move data to the selected device (CPU or GPU)
            (batchX, batchY) = (batchX.to(DEVICE), batchY.to(DEVICE))

            # Zero the parameter gradients
            opt.zero_grad()

            # Forward pass: compute model output
            outputs = model(batchX)

            # Compute the loss
            loss = loss_func(outputs, batchY.long())

            # Backward pass: compute gradients
            loss.backward()

            # Optimize the model
            opt.step()

            # Update stats
            running_loss += loss.item() * batchX.size(0)  # Accumulate loss
            _, predicted = torch.max(outputs, 1)
            total += batchY.size(0)
            correct += (predicted == batchY.long()).sum().item()

        # Compute the average loss and accuracy for the epoch
        epoch_loss = running_loss / len(trainX)
        accuracy = correct / total
        print(f"Epoch: {epoch + 1}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")

# ================ Evaluation =====================
# initialize tracker variables for testing, then set our model to
# evaluation mode

def test_mlp(model, loss_fun, num_epochs):
    testLoss = 0
    testAcc  = 0
    samples  = 0
    model.eval()
    print('------------ TESTING ANN ----------------')
    with torch.no_grad():
        for epoch in range(0, num_epochs):
            # loop over the current batch of test data
            for (batchX, batchY) in gen_next_batch(testX, testY, BATCH_SIZE):
                (batchX, batchY) = (batchX.to(DEVICE), batchY.to(DEVICE))
                (batchX, batchY) = (batchX.to(DEVICE), batchY.to(DEVICE))

                # Forward pass: compute model output
                outputs = model(batchX)

                # Compute the loss
                loss = loss_func(outputs, batchY.long())
                test_loss += loss.item() * batchX.size(0)

                # Compute accuracy
                _, predicted = torch.max(outputs, 1)
                total += batchY.size(0)
                correct += (predicted == batchY.long()).sum().item()

            # Compute and print average loss and accuracy for the current epoch
            avg_loss = test_loss / len(testX)
            accuracy = correct / total
            print(f"Epoch: {epoch + 1}, Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")

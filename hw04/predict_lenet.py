############################################
# module: predict_lenet.py
#############################################

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
import argparse
import imutils
import torch
import cv2

np.random.seed(13)

# let's use arg parser to parse args
ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', type=str, required=True,
	        help='path to the trained PyTorch model')
args = vars(ap.parse_args())

# set the device we will be using to test the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load the KMNIST dataset and randomly grab 10 data points
print('<DBG> loading KMNIST test dataset...')
testData = KMNIST(root='data', train=False, download=True,
	          transform=ToTensor())
idxs = np.random.choice(range(0, len(testData)), size=(20,))
testData = Subset(testData, idxs)
# initialize the test data loader
testDataLoader = DataLoader(testData, batch_size=1)
# load the model and set it to evaluation mode
model = torch.load(args['model']).to(device)
model.eval()

# switch off autograd
with torch.no_grad():
    # loop over the test set
    for (image, label) in testDataLoader:
        # grab the original image and ground truth label
        orig_img = image.numpy().squeeze(axis=(0, 1))
        gt_lbl = testData.dataset.classes[label.numpy()[0]]
        # send the input to the device and make predictions on it
        image = image.to(device)
        pred  = model(image)
        # find the class label index with the largest corresponding
        # probability
        idx = pred.argmax(axis=1).cpu().numpy()[0]
        pred_lbl = testData.dataset.classes[idx]
	# convert the image from grayscale to RGB (so we can draw on
	# it) and resize it (so we can more easily see it on our
	# screen)
        orig_img = np.dstack([orig_img] * 3)
        orig_img = imutils.resize(orig_img, width=128)
        # draw the predicted class label on it
        color = (0, 255, 0) if gt_lbl == pred_lbl else (0, 0, 255)
        cv2.putText(orig_img, gt_lbl, (2, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2)
        # display the result in terminal and show the input image
        print('<DBG>: ground truth: {}, predicted: {}'.format(gt_lbl, pred_lbl))
        cv2.imshow('Character', orig_img)
        cv2.waitKey(0)

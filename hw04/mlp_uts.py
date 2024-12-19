
####################################################
# CS 5600/6600: F24: Assignment 4: Unit Tests
# bugs to valdimir kulyukin in canvas.
#####################################################

import numpy as np
from mlp_hw import *
from  train_eval_mlp_hw import *
import unittest

'''
One of my sample outputs:
~/teaching/AI/F24/hw/hw04$ python mlp_uts.py 
<DBG>: training using cuda...
<DBG>: preparing data...
------------ TRAINING ANN ----------------
<DBG>: epoch: 1...
epoch: 1 train loss: 0.685 train accuracy: 0.804
<DBG>: epoch: 2...
epoch: 2 train loss: 0.429 train accuracy: 0.839
<DBG>: epoch: 3...
epoch: 3 train loss: 0.323 train accuracy: 0.839
<DBG>: epoch: 4...
epoch: 4 train loss: 0.226 train accuracy: 1.000
<DBG>: epoch: 5...
epoch: 5 train loss: 0.141 train accuracy: 1.000
.------------ TRAINING ANN ----------------
<DBG>: epoch: 1...
epoch: 1 train loss: 1.127 train accuracy: 0.732
<DBG>: epoch: 2...
epoch: 2 train loss: 0.735 train accuracy: 0.946
<DBG>: epoch: 3...
epoch: 3 train loss: 0.474 train accuracy: 1.000
<DBG>: epoch: 4...
epoch: 4 train loss: 0.290 train accuracy: 1.000
<DBG>: epoch: 5...
epoch: 5 train loss: 0.168 train accuracy: 1.000
.------------ TRAINING ANN ----------------
<DBG>: epoch: 1...
epoch: 1 train loss: 1.543 train accuracy: 0.286
<DBG>: epoch: 2...
epoch: 2 train loss: 1.378 train accuracy: 0.518
<DBG>: epoch: 3...
epoch: 3 train loss: 1.166 train accuracy: 0.839
<DBG>: epoch: 4...
epoch: 4 train loss: 0.906 train accuracy: 1.000
<DBG>: epoch: 5...
epoch: 5 train loss: 0.646 train accuracy: 1.000
.------------ TRAINING ANN ----------------
<DBG>: epoch: 1...
epoch: 1 train loss: 1.523 train accuracy: 0.357
<DBG>: epoch: 2...
epoch: 2 train loss: 1.412 train accuracy: 0.625
<DBG>: epoch: 3...
epoch: 3 train loss: 1.323 train accuracy: 0.661
<DBG>: epoch: 4...
epoch: 4 train loss: 1.247 train accuracy: 0.696
<DBG>: epoch: 5...
epoch: 5 train loss: 1.181 train accuracy: 1.000
.------------ TRAINING ANN ----------------
<DBG>: epoch: 1...
epoch: 1 train loss: 1.707 train accuracy: 0.107
<DBG>: epoch: 2...
epoch: 2 train loss: 1.707 train accuracy: 0.107
<DBG>: epoch: 3...
epoch: 3 train loss: 1.707 train accuracy: 0.107
<DBG>: epoch: 4...
epoch: 4 train loss: 1.707 train accuracy: 0.107
<DBG>: epoch: 5...
epoch: 5 train loss: 1.707 train accuracy: 0.107
.------------ TRAINING ANN ----------------
<DBG>: epoch: 1...
epoch: 1 train loss: 1.650 train accuracy: 0.161
<DBG>: epoch: 2...
epoch: 2 train loss: 1.650 train accuracy: 0.161
<DBG>: epoch: 3...
epoch: 3 train loss: 1.650 train accuracy: 0.161
<DBG>: epoch: 4...
epoch: 4 train loss: 1.650 train accuracy: 0.161
<DBG>: epoch: 5...
epoch: 5 train loss: 1.650 train accuracy: 0.161
.
----------------------------------------------------------------------
Ran 6 tests in 0.964s
OK
'''

class mlp_uts(unittest.TestCase):

    def test_mlp_relu_1(self, num_epochs=5):
        train_mlp(ReLU_MLP1, ReLU_OPT1, lossFunc, num_epochs)

    def test_mlp_relu_2(self, num_epochs=5):
        train_mlp(ReLU_MLP2, ReLU_OPT2, lossFunc, num_epochs)

    def test_mlp_relu_3(self, num_epochs=5):
        train_mlp(ReLU_MLP3, ReLU_OPT3, lossFunc, num_epochs)

    def test_mlp_sig_1(self, num_epochs=5):
        train_mlp(Sig_MLP1, Sig_OPT1, lossFunc, num_epochs)

    def test_mlp_sig_2(self, num_epochs=5):
        train_mlp(Sig_MLP2, ReLU_OPT2, lossFunc, num_epochs)

    def test_mlp_sig_3(self, num_epochs=5):
        train_mlp(Sig_MLP3, ReLU_OPT3, lossFunc, num_epochs)

if __name__ == '__main__':
    unittest.main()


 
    
        
        

    
        






import numpy as np
from typing import List

class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        n = len(y_true)
        z1 = np.asarray(W1) @ np.asarray(x)   + b1
        a1 = np.maximum(z1, 0)
        z2 =  np.asarray(W2) @ a1 + b2
        loss = np.mean((z2-y_true)**2)
        dz2 = 2*(z2-y_true)/n
        dW2 = np.outer(dz2.T, a1)
        db2 = dz2
        da1 = dz2 @ W2
        dz1 =  da1*(z1>0)
        dW1 = np.outer(dz1.T, np.asarray(x))
        db1 = dz1
        return {
                "loss": np.round(loss,4), 
                "dW1" : np.round(dW1,4), 
                "db1": np.round(db1,4), 
                "dW2": np.round(dW2,4), 
                "db2": np.round(db2,4)
            }


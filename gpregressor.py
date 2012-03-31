import random as random
import numpy as np
import math
import scipy.linalg

class Kernal(object):
    matrix= np.matrix(float)
    Lambda = 1

    def __init__(self, Lambda = None):
        if Lambda is None:
            Lambda =1
        else:
            self.Lambda = Lambda

    def compute_kernal_matrix(self,x,x2):
        """Computes the kernal matrix between two given x vectors, using a gaussian RBF kernal"""
        d = np.matrix(np.sum(np.abs(x)**2,axis=1))
        d2 = np.matrix(np.sum(np.abs(x2)**2,axis=1))
        ones_d = np.ones_like(d)
        ones_d2 = np.ones_like(d2)
        X = np.matrix(x)
        X2 = np.matrix(x2)
        #sq_norms=d.transpose()*(ones_d) + ones_d.transpose()*d - 2*X*X.transpose()
        sq_norms=d.transpose()*(ones_d) + ones_d2.transpose()*d2 - 2*X*X2.transpose()
        return np.exp(-sq_norms/(self.Lambda**2))   # Using Gaussian RBF Kernal


    def store_kernal_matrix(self,x,x2):
        self.matrix = self.compute_kernal_matrix(x,x2)
    
    def get_kernal_matrix(self):
        return self.matrix

    def set_Lambda(self,s):
        self.Lambda = s



class GPRegressor(object):
    """Performs Gaussian Process Regression using a given Kernal"""
    quantiles = []
    sigma = [0.01, 0.03, 0.1, 0.3, 1., 3.]
    kernal = Kernal()
    
    def approximate_quantiles(self,x, num_samples=1000):
        """Randomly samples 1000 x pairs in order to compute various data quantiles.
           This is used for calibrating the kernal."""
        p1= random.sample(x,num_samples)
        p2= random.sample(x,num_samples)
        # Compute the array of norms
        norms = [np.linalg.norm(a-b) for a,b in zip(p1,p2)]
        # Sort the norms
        norms.sort()
        quant_index = [ int(math.floor(len(norms)*quant))-1 for quant in [0.05,0.1,0.5,0.9,0.95]]
        self.quantiles = [ norms[ind] for ind in quant_index]

    def estimate_mean(X,y,Xtest,sigma,Lambda):
        I = np.eye(len(X))
        K = self.kernal.get_kernal_matrix()
        L = scipy.linalg.cho_factor(K + (sigma**2)*I)
        LT = (L[0].transpose(),not L[1])
        a = scipy.linalg.cho_solve(L,y)
        alpha = scipy.linalg.cho_solve(LT,y)
        

        


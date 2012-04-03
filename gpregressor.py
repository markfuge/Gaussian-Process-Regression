import random as random
import numpy as np
import math
import scipy.linalg
import itertools

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

    def get_kernal_partions(self,partion_number,partion_indices,remainder=False):
        """Returns a set of partions of the kernal matrix.
           Given a partion of indices P*, this function returns 4 matrices:
           K(P*,P*),K(P*,P),K(P,P*),K(P,P)
           where P represents the remaining partions
        """
        #partion_indices = self.partion_indices  # copy the partions
        pi = list(partion_indices)
        target_inds = pi[partion_number]
        pi.pop(partion_number)
        # Now add the rest of the indices together
        remain_inds = list(itertools.chain(*pi))
        ix1 = np.ix_(target_inds,target_inds)
        ix2 = np.ix_(target_inds,remain_inds)
        ix3 = np.ix_(remain_inds,target_inds)
        ix4 = np.ix_(remain_inds,remain_inds)
        K = self.matrix
        return (K[ix1],K[ix2],K[ix3],K[ix4])

    def store_kernal_matrix(self,x,x2):
        self.matrix = self.compute_kernal_matrix(x,x2)
    
    def get_kernal_matrix(self):
        return self.matrix

    def set_Lambda(self,s):
        self.Lambda = s



class GPRegressor(object):
    """Performs Gaussian Process Regression using a given Kernal"""
    quantiles = []
    X=[]
    y=[]
    kernal = Kernal()
    partion_indices = []
    
    def __init__(self,X,y,num_partions=5):
        self.X=X
        self.y=y
        self.num_partions=num_partions
        self.approximate_quantiles(X)
        #self.init_kernal(X,y)
        self.partion_data_indices(self.num_partions)

    def reset_lambda(self,Lambda):
        self.kernal.Lambda = Lambda
        self.init_kernal(self.X,self.y)

    def partion_data_indices(self,num_partions=None):
        if num_partions is None:
            num_partions = self.num_partions
        N = int(self.X.shape[0])   # total number of rows
        n = N/num_partions
        indices = range(0,N)
        for i in range(0,num_partions-1):
            self.partion_indices.append(indices[n*i:n*(i+1)])
        self.partion_indices.append(indices[n*(i+1):])

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
    
    def return_quantiles(self):
        return self.quantiles

    def init_kernal(self,X,y):
        self.kernal.store_kernal_matrix(X,X)

    def get_y_partions(self,partion_number,partion_indices):
        pi = list(partion_indices)
        target_inds = pi[partion_number]
        pi.pop(partion_number)
        # Now add the rest of the indices together
        remain_inds = list(itertools.chain(*pi))
        return (self.y[target_inds],self.y[remain_inds])

    def estimate_mean(self,partion_number,sigma,Lambda):
        #K = self.kernal.get_kernal_matrix()
        # Get the desired partions of the kernal matrix
        KTKT,KTK,KKT,KK = self.kernal.get_kernal_partions(partion_number,self.partion_indices)
        yT,y = self.get_y_partions(partion_number,self.partion_indices)
        # Generate the additional noise diagonal matrix
        I = np.eye(KK.shape[0])
        # Factorize the non-test kernal matrix, with the added noise term
        L = scipy.linalg.cho_factor(KK + (sigma**2)*I)
        LT = (L[0].transpose(),not L[1])
        a = scipy.linalg.cho_solve(L,y)
        alpha = scipy.linalg.cho_solve(LT,y)

        # Now to evaluate the estimates
        fT = KTK*alpha  # Estimated means on the validation data
        validation_error = 0.5*np.square(yT-fT)
        v_error_per = np.sum(validation_error)/int(yT.shape[0]) # get the error, per example

        f = KK*alpha  # Estimated means on the training data
        training_error = 0.5*np.square(y-f)
        t_error_per = np.sum(training_error)/int(y.shape[0])

        return v_error_per,t_error_per

        

        


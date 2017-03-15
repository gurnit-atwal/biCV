from __future__ import division 

import numpy as np 
import sklearn as sk 
from sklearn.decomposition import NMF
import scipy.sparse as sp 
from scipy.optimize import nnls
from math import sqrt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import randomized_svd, safe_sparse_dot, squared_norm


def norm(x): 
    return sqrt(squared_norm(x))
    
def trace_dot(X, Y): 
    return np.dot(X.ravel(), Y.ravel())
    
def _sparesness(x): 
    """ Hoyer's measure of sparsity for a vector """ 
    sqrt_n = sqrt(len(x))
    return (sqrt_n - np.linalg.norm(x,1) / norm(x)) / (sqrt_n -1)


n = 10
m = 11
crossVal = 3


X = np.zeros((n,m), dtype= np.int)

K = np.arange(1, np.amin(n,m) + 1)
I = np.arange(1, m +1)
J = np.arange(1, n + 1)

L = np.arange(1,crossVal)

BCV = [0]*len(K) 
 

def _nls_subproblem(X, W, H, tol, max_iter, sigma = 0.01, beta =0.1): 
    #solves for H given constant W, X
    
    WtX = safe_sparse_dot(W.T, X)
    WtW = np.dot(W.T, W)
    
    alpha = 1 
    for iter in range(1, max_iter + 1): 
        grad = np.dot(WtW, H) - WtX
        
        if norm(grad * np.logical_or(grad <0, H > 0)) < tol: 
            break 
    
        Hp = H 
        
        for inner_iter in range(19): 
            Hn = H - alpha * grad 
            Hn *= Hn > 0
            d = Hn - H 
            gradd = np.dot(grad.ravel(), d.ravel())
            dQd = np.dot(np.dot(WtW, d).ravel(), d.ravel())
            suff_decr = (1-sigma) * grad + 0.5 * dQd < 0
            if inner_iter == 0: 
                decr_alpha = not suff_decr 
                
            if decr_alpha: 
                if suff_decr: 
                    H = Hn 
                    break 
                else: 
                    alpha *= beta 
            elif not suff_decr or (Hp==Hn).all(): 
                H = Hp 
                break 
            else: 
                alpha /= beta 
                Hp = Hn 
        
    return H 

#Create 10 fold before starting 
#For columns, have 

def cvIndices(index, fold = 10): 
    test = [] 
    train = []
    kf = KFold(n_splits = fold)
    for j in kf.split(index): 
        test.append(j[1])
        train.append(j[0])
    return train, test  

trainI, testI = cvIndices(m)
trainJ, testJ = cvIndices(n)

for fold in range(10): 
    I_l = testI[fold]
    J_L = testJ[fold]
    LeftInI = trainI[fold]
    LeftInJ = trainJ[fold]
    for k in K: 
        #remove I_l, J_l from X 
        
        leftOutI = X[I_l, :]
        leftOutJ = X[:, J_l] 
        #XnegI = np.delete(X, I_l, axis = 0) 
        #XnegJ = np.delete(X, J_l, axis = 1)
        XnegI = np.delete(leftOutJ, J_l, axis = 0)
        XnegJ = np.delete(leftOutI, I_l, axis = 1)
        #XnegInegJ = np.delete(X, I_l, axis = 0)
        #XnegInegJ = np.delete(XnegInegJ, J_l, axis = 1)
        XnegInegJ = X[leftInI,LeftInJ]
        Xij = X[I_l, J_l]
        
        #NMF on X-J-I, store those W and H 
        
        nmf = NMF(n_components = k, nls_max_iter = 2000)
        WnegIJ = nmf.fit_transform(XnegInegJ)
        HnegIJ = nmf.components_ 
        
        #Solve WiJ 
        Wij = _nls_subproblem(XnegI.T, H.T, WnegIJ.T)
        
        #Solve HiJ 
        Hij = _nls_subproblem(XnegJ, W, HnegIJ, tolH=1e-4, max_iter = 2000) 
        
        #Xij = np.doct(WiJ, HiJ) 
        
        Xijestimate = np.dot(Wij, Hij)
        
        error = squared_norm(Xij, Xijestimate)
        #error = frobenius(X, Xij)
        BCV[k] += error 
        
rank = np.argmin(BCV)


def biCrossValidation(X, total = 10):
    m,n = X.shape
    K = np.arange(1, np.amin(n,m))
    trainI, testI = cvIndices(m, total)
    trainJ, testJ = cvIndices(n, total)
    
    for fold in range(total): 
        I_l = testI[fold]
        J_L = testJ[fold]
        LeftInI = trainI[fold]
        LeftInJ = trainJ[fold]
        for k in K: 
        #remove I_l, J_l from X 
        
            leftOutI = X[I_l, :]
            leftOutJ = X[:, J_l] 
        #XnegI = np.delete(X, I_l, axis = 0) 
        #XnegJ = np.delete(X, J_l, axis = 1)
            XnegI = np.delete(leftOutJ, J_l, axis = 0)
            XnegJ = np.delete(leftOutI, I_l, axis = 1)
        #XnegInegJ = np.delete(X, I_l, axis = 0)
        #XnegInegJ = np.delete(XnegInegJ, J_l, axis = 1)
            XnegInegJ = X[leftInI,LeftInJ]
            Xij = X[I_l, J_l]
        
        #NMF on X-J-I, store those W and H 
        
            nmf = NMF(n_components = k, nls_max_iter = 2000)
            WnegIJ = nmf.fit_transform(XnegInegJ)
            HnegIJ = nmf.components_ 
        
        #Solve WiJ 
            Wij = _nls_subproblem(XnegI.T, H.T, WnegIJ.T)
        
        #Solve HiJ 
            Hij = _nls_subproblem(XnegJ, W, HnegIJ, tolH=1e-4, max_iter = 2000) 
        
        #Xij = np.doct(WiJ, HiJ) 
        
            Xijestimate = np.dot(Wij, Hij)
        
            error = squared_norm(Xij, Xijestimate)
        #error = frobenius(X, Xij)
            BCV[k] += error 
    return np.argmin(BCV)

rank = biCrossValidation(X, total = 10)

    
    
    
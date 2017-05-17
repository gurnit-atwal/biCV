from __future__ import division 

import numpy as np 
import sklearn as sk 
from sklearn.decomposition import NMF
import scipy.sparse as sp 
from scipy.optimize import nnls
from math import sqrt

from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import randomized_svd, safe_sparse_dot, squared_norm

from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns

def norm(x): 
    return sqrt(squared_norm(x))
    
def trace_dot(X, Y): 
    return np.dot(X.ravel(), Y.ravel())
    
def _sparesness(x): 
    """ Hoyer's measure of sparsity for a vector """ 
    sqrt_n = sqrt(len(x))
    return (sqrt_n - np.linalg.norm(x,1) / norm(x)) / (sqrt_n -1)
    

def _nls_subproblem(X, W,H, tol = 1e-4, max_iter = 2000, sigma = 0.01, beta= 0.1): 
    #nn-leastsquares using projected gradient descent 
    
    WtX = safe_sparse_dot(W.T, X)
    WtW = np.dot(W.T, W)
    alpha = 1 
    for iter in range(1, max_iter + 1): 
        grad = np.dot(WtW, H) - WtX
        
        if norm(grad * np.logical_or(grad<0, H>0)) < tol: 
            break 
    Hp = H 
    
    for inner_iter in range(19): 
        Hn = H - alpha * grad
        Hn *= Hn > 0 
        d = Hn - H 
        gradd = np.dot(grad.ravel(), d.ravel())
        dQd = np.dot(np.dot(WtW, d).ravel(), d.ravel())
        suff_decr = (1-sigma) * gradd + 0.5 * dQd <0 
        if inner_iter == 0: 
            decr_alpha = not suff_decr 
        
        if decr_alpha: 
            if suff_decr: 
                H = Hn 
                break 
            else: 
                alpha *= beta 
        elif not suff_decr or (Hp ==Hn).all(): 
            H = Hp 
            break 
        else: 
            alpha /= beta 
            Hp = Hn 
    return H 
    
def cvIndices(index, fold = 3): 
    test = [] 
    train = []
    kf = KFold(n_splits = fold)
    for j in kf.split(range(index)): 
        test.append(j[1])
        train.append(j[0])
    return train, test  

K = np.arange(3, 13)
BCV = np.zeros(len(K))    
#X = np.random.random((1779,16))
X = np.load('/Users/gurnitatwal/Desktop/ProjectStuff/rearrSigs/rearrSigs/rearrCorrectMatrix.npy')
m = X.shape[1]
n = X.shape[0]
trainI, testI = cvIndices(n)
trainJ, testJ = cvIndices(m)

'''
for k in K: 
#for fold in range(10): 
    sc = 0
    for fold in range(10):
        I_l = testI[fold]
        J_l = testJ[fold]
        LeftInI = trainI[fold]
        LeftInJ = trainJ[fold]
        LeftOutI = X[I_l,:]
        LeftOutJ = X[:, J_l]
        XnegI = np.delete(LeftOutI, J_l, axis = 1)
        XnegJ = np.delete(LeftOutJ, I_l, axis = 0)
        
        XnegINegJ = X[LeftInI,:]
        XnegINegJ = XnegINegJ[:, LeftInJ]
        
        Xij = X[I_l,:]
        Xij = Xij[:,J_l]
        
        nmf = NMF(n_components = k, nls_max_iter = 2000)
        WnegIJ = nmf.fit_transform(XnegINegJ)
        HnegIJ = nmf.components_
        
        W = np.random.random((len(I_l), k))
        
        Wij = _nls_subproblem(XnegI.T, HnegIJ.T, W.T, tol = 1e-4, max_iter = 2000)
        
        H = np.random.random((k, len(J_l)))
        
        Hij = _nls_subproblem(XnegJ,WnegIJ, H)
        
        Xijest = np.dot(Wij.T, Hij)
        err = Xij - Xijest
        #error = squared_norm(Xij, Xijest)
        error = squared_norm(err)
        sc += error
    BCV[k-3] += error / 10 
rank = np.argmin(BCV) + 3
BCV = BCV / 10
r = np.amin(np.argwhere(BCV <= np.amin(BCV) + np.std(BCV)))

print r+3, rank
'''

def biCrossValidation(X, num, total = 10, minRank = 3):
    #K = np.arange(minRank, num)
    K = (5, 10,15,20, 50)
    BCV = np.zeros(len(K))    
    #X = np.random.random((15,15))
    m = X.shape[1]
    n = X.shape[0]
    trainI, testI = cvIndices(n)
    trainJ, testJ = cvIndices(m)

    for i in range(len(K)): 
#for fold in range(10): 
        k = K[i]
        sc = 0
        for fold in range(3):
            I_l = testI[fold]
            J_l = testJ[fold]
            LeftInI = trainI[fold]
            LeftInJ = trainJ[fold]
            LeftOutI = X[I_l,:]
            LeftOutJ = X[:, J_l]
            XnegI = np.delete(LeftOutI, J_l, axis = 1)
            XnegJ = np.delete(LeftOutJ, I_l, axis = 0)
        
            XnegINegJ = X[LeftInI,:]
            XnegINegJ = XnegINegJ[:, LeftInJ]
        
            Xij = X[I_l,:]
            Xij = Xij[:,J_l]
        
            nmf = NMF(n_components = k, nls_max_iter = 2000)
            WnegIJ = nmf.fit_transform(XnegINegJ)
            HnegIJ = nmf.components_
        
            W = np.random.random((len(I_l), k))
        
            Wij = _nls_subproblem(XnegI.T, HnegIJ.T, W.T, tol = 1e-4, max_iter = 2000)
        
            H = np.random.random((k, len(J_l)))
        
            Hij = _nls_subproblem(XnegJ,WnegIJ, H)
        
            Xijest = np.dot(Wij.T, Hij)
            err = Xij - Xijest
        #error = squared_norm(Xij, Xijest)
            error = squared_norm(err)
            sc += error
        BCV[i] += error / 10 
    #rank = np.argmin(BCV) + 3
    BCV = BCV / 10
    r = np.amin(np.argwhere(BCV <= np.amin(BCV) + np.std(BCV)))
    
    ## I should add STD to np.argmin(BCV) and then recheck for argmin. 
    ## If np.argmin(BCV) returns a tuple, take the earliest entry 
    return np.argmin(BCV)

values = [] 
K = (5, 10,15,20, 50)
for i in range(100): 
    a = biCrossValidation(X, num = 13)
    values.append(a)
    print K[a]


best = [K[x] for x in values]
best = np.array(best)
np.save('/scratch/q/qmorris/gurnit/pcawg/pwgs_results/cvResults.npy', best, allow_pickle=False)

g = sns.distplot(best)
g.set_title('Distribution of Optimal Topic Number for 200 Rounds Bi-Cross-Validation')
g.set_xlabel('Number of Topics')
g.set_ylabel('Proportion')
g.figure.savefig('/Users/gurnitatwal/Desktop/biCVPlot.png')
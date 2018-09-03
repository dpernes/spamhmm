import numpy as np
import math
from hmmlearn.utils import iter_from_X_lengths

def relu(x):
  return x*(x > 0)


def drelu(x):
  return 1*(x > 0)


def relu_normalization(x, axis=0):
  num = relu(x)**2
  z = relu(x)**2
  return num / np.expand_dims(z.sum(axis), axis)
  

def reg_relu(G, theta):
  K, M = theta.shape
  alpha = relu_normalization(theta, axis=1)
  r = 0.
  
  for k in range(K):
    for kk in range(K):
      if kk == k:
        continue
      
      r += 0.5 * G[k, kk] * np.dot(alpha[k,:], alpha[kk,:])
  
  return r


def lst2np(Xlist, ylist):
  X = []
  y = []
  lengths = []
  for ap, seq in zip(ylist, Xlist):
    X.append(seq)
    y.append(ap)
    lengths.append(seq.shape[0])
  
  X = np.concatenate(X)
  y = np.array(y)
  
  return X, y, lengths


def np2lst(Xnp, ynp, lengths):
  X = []
  y = np.array(ynp)
  
  for (i,j) in iter_from_X_lengths(Xnp, lengths):
    X.append(Xnp[i:j, :])
    
  return X, y  


# def get_S_1HMM(K, M, S, D):
#   par_mhmm = K*(M-1) + M*(S-1) + M*S*(S-1) + 2*M*S*D
  
#   for s in range(1000):
#     par_1hmm = s-1 + s*(s-1) + 2*s*D
#     if par_1hmm >= par_mhmm:
#       return s

def get_S_1HMM(K, M, S, D):
  s = int(math.ceil(S * M**0.5))
  
  return s

      
# def get_S_KHMM(K, M, S, D):
#   par_mhmm = K*(M-1) + M*(S-1) + M*S*(S-1) + 2*M*S*D
  
#   for s in range(1000):
#     par_khmm = K*(s-1) + K*s*(s-1) + 2*K*s*D
#     if par_khmm >= par_mhmm:
#       return s


def get_S_KHMM(K, M, S, D):
  s = int(math.ceil(S * (1.*M/K)**0.5))
  
  return s
  
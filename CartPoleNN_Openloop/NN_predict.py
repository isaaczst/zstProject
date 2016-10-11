# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 11:48:55 2016

@author: a8zn5
"""

import numpy as np

def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

###########################################


def sigmoidNN(state, action):
    
    #format the sample and label
    sampleRange = state.shape[1] - 3
    X = np.array((15,sampleRange))
    Y = np.array((4,sampleRange))
    for i in range(sampleRange):   
        for j in range(3): #+0, +1, +2
            X[(i*5+j):(i*5+j+4),i] = state[:,i+j] # +0, +1, +2, +3
            X[(i*5+j+4),i] = action[i+j] #+4
        Y[:,i] = state[:,i+3] #+3

    for k in xrange(10): #stochastic/minibatch way, no need too many times
    
        # Feed forward through layers 0, 1, and 2
        l0 = X.T # 15 x sampleNo.
        l1 = nonlin(np.dot(l0,syn0)) # sampleNo x 15 x 15 x 30
        l2 = nonlin(np.dot(l1,syn1)) # sampleNo x 30 x 30 x 4
    
        # how much did we miss the target value?
        l2_error = Y.T - l2
    
        if k == 9: #(j% 10000) == 0:
            #print "Error:" + str(np.mean(np.abs(l2_error)))
            accuracy_rate = np.mean(np.abs(l2_error))
    
        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        l2_delta = l2_error*nonlin(l2,deriv=True)
    
        # how much did each l1 value contribute to the l2 error (according to the weights)?
        l1_error = l2_delta.dot(syn1.T)
    
        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        l1_delta = l1_error * nonlin(l1,deriv=True)
    
        syn1 += l1.T.dot(l2_delta)
        syn0 += l0.T.dot(l1_delta)
        
    return syn0, syn1, accuracy_rate
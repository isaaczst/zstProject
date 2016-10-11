# -*- coding: utf-8 -*-
"""
Created on Sun Oct 02 14:05:18 2016

@author: a8zn5
"""
import numpy as np


class nnLearner1(object):
    def __init__(self, structure,timeStepRange):
        self.l0Size=structure[0] # inputs
        self.l1Size=structure[1]  # neurons for hidden layer1
        self.l2Size=structure[2]  # neurons for hidden layer2
        self.l3Size=structure[3]  # neurons for output layer
        np.random.seed(1)
        self.syn0 = 0.2*np.random.random((self.l0Size,self.l1Size)) - 0.1  
        self.syn1 = 0.2*np.random.random((self.l1Size,self.l2Size)) - 0.1 
        self.syn2 = 0.2*np.random.random((self.l2Size,self.l3Size)) - 0.1 
        self.bias0=0.2*np.random.random((self.l1Size,1)).T - 0.1 # next layer's neuron quantity
        self.bias1=0.2*np.random.random((self.l2Size,1)).T - 0.1 
        self.bias2=0.2*np.random.random((self.l3Size,1)).T - 0.1 
        self.alpha = 0.01

    def sigmoid(self, x, deriv=False):
        if(deriv==True):
            return x*(1-x)
        return 1/(1+np.exp(-x))
    
    def linAmp(self, x, deriv=False):
        if(deriv==True):
            constDeriv=np.ones(x.shape)
            return constDeriv
        return x   


    def train(self, state, action):
        sampleSize=state.shape[0]-1
        learningRate=self.alpha/sampleSize
        X = np.zeros((sampleSize, self.l0Size))
        Y = np.zeros((sampleSize, self.l3Size))
        
        X[:, :state.shape[1]] = state[:sampleSize, :] #0~range-1
        X[:, state.shape[1]:]=action[:sampleSize, :] #0~range-1
        Y = state[1:,:] # 1~range

        repetitionNo=1000
        for k in xrange(repetitionNo): #stochastic/minibatch way, no need too many times    
            # Feed forward through layers 0, 1, and 2
            l0 = X # sampleRange x l0Size
            l1 = self.sigmoid(np.dot(l0,self.syn0)+self.bias0) # sampleRange x l0Size x l0Size x l1Size
            l2 = self.sigmoid(np.dot(l1,self.syn1)+self.bias1) # sampleRange x l1Size x l1Size x l2Size
            l3 = self.linAmp(np.dot(l2,self.syn2)+self.bias2) # sampleRange x l2Size x l2Size x l3Size    
            l3_error =  (Y - l3)          
            if k == repetitionNo-1: #(j% 10000) == 0:
                errorRate = np.mean(np.abs(l3_error))
            # below backpropagation 
            l3_delta = l3_error*self.linAmp(l3,deriv=True) # delta is the derivative of prediction error w.r.t. layer input
            l2_error = l3_delta.dot(self.syn2.T) # error means error-term
            l2_delta = l2_error*self.sigmoid(l2,deriv=True)
            l1_error = l2_delta.dot(self.syn1.T)    
            l1_delta = l1_error * self.sigmoid(l1,deriv=True)

            self.syn2 += l2.T.dot(l3_delta)*learningRate # multiplication of matrices cause the summation, so need average
            self.syn1 += l1.T.dot(l2_delta)*learningRate
            self.syn0 += l0.T.dot(l1_delta)*learningRate
            self.bias0 += np.sum(l1_delta,axis=0)*learningRate # Compute sum of each column;
            self.bias1 += np.sum(l2_delta,axis=0)*learningRate 
            self.bias2 += np.sum(l3_delta,axis=0)*learningRate 
        
        return  l3, errorRate
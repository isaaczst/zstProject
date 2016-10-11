# -*- coding: utf-8 -*-
"""
Created on Mon Oct 03 00:49:43 2016

@author: a8zn5
"""



class CartPoleV0(object):
    def __init__(self, parameter):
        self.gravity = 9.8
        self.masscart = parameter[0]
        self.masspole = parameter[1]
        self.total_mass = (self.masspole + self.masscart)
        self.length = parameter[2] # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = parameter[3]
        self.tau = 0.02  # seconds between state updates
        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        #self.x_threshold = parameter[0]     
     
    def step(self, state, force):

        x, x_dot, theta, theta_dot = state
        #    force = force_mag if action==1 else -force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x  = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        state = (x,x_dot,theta,theta_dot)

        return np.array(state) #, reward, done, {}
        
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
        self.bias0=0.2*np.random.random((self.l0Size,1)) - 0.1 
        self.bias1=0.2*np.random.random((self.l1Size,1)) - 0.1 
        self.bias2=0.2*np.random.random((self.l2Size,1)) - 0.1 
        self.alpha = 0.1

    def nonlin(x,deriv=False):
        if(deriv==True):
            return x*(1-x)
        return 1/(1+np.exp(-x))
    
    def linAmp(x,deriv=False):
        if(deriv==True):
            constDeriv=np.ones(x.shape)
            return constDeriv
        return x   


    def train(self, state, action):
        X = np.zeros((state.shape[0]-1,self.l0Size))
        Y = np.zeros((state.shape[0]-1,self.l3Size))
        
        X[:state.shape[0], :state.shap[1]] = state[:state.shape[0], :] #0~range-1
        X[:state.shape[0], state.shap[1]]=action[:state.shape[0], :] #0~range-1
        Y = state[1:,:] # 1~range

        repetitionNo=100
        for k in xrange(repetitionNo): #stochastic/minibatch way, no need too many times    
            # Feed forward through layers 0, 1, and 2
            l0 = X # sampleRange x l0Size
            l1 = self.nonlin(np.dot(l0,self.syn0)) # sampleRange x l0Size x l0Size x l1Size
            l2 = self.nonlin(np.dot(l1,self.syn1)) # sampleRange x l1Size x l1Size x l2Size
            l3 = self.linAmp(np.dot(l2,self.syn2)) # sampleRange x l2Size x l2Size x l3Size    
            l3_error = Y - l3
            # below backpropagation
            if k == repetitionNo-1: #(j% 10000) == 0:
                errorRate = np.mean(np.abs(l3_error))

            l3_delta = l3_error*self.linAmp(l3,deriv=True)
            l2_error = l3_delta.dot(self.syn2.T)
            l2_delta = l2_error*self.nonlin(l2,deriv=True)
            l1_error = l2_delta.dot(self.syn1.T)    
            l1_delta = l1_error * self.nonlin(l1,deriv=True)

            self.syn2 += l2.T.dot(l3_delta)*self.alpha
            self.syn1 += l1.T.dot(l2_delta)*self.alpha
            self.syn0 += l0.T.dot(l1_delta)*self.alpha
        
        return  l3, errorRate

class taskLocal1(object):
    
    def __init__(self,timeStepRange):
        self.timeStepRange=timeStepRange
        self.state=np.zeros((timeStepRange,4))
        self.state_hat=np.zeros((timeStepRange,4))
        
    def reset(self, plant):
        plant.state=np.array([0,0,0,0])
        
    def genAction(self,i_episode, plant):
        self.action=10*np.random.random((self.timeStepRange,1)) - 5  
        
    def step(self,plant):
        for i in range(self.timeStepRange-1):
            self.state[i+1,:]=plant.step(self.state[i,:], self.action[i])
        
    def evaluation(self,i_episode,nn):
        self.state_hat[1:, :], errorRate = nn.train(self.state,self.action)
        if i_episode %100: 
            print errorRate


    def plotFig(self):
        # Plot the points using matplotlib
        timeRange=self.state.shape[0]
        t=range(timeRange)
        #
        plt.plot(t, self.action[t,0])
        plt.xlabel(' ')
        plt.ylabel('N')
        plt.title(' ')
        plt.legend(['force'])   
        plt.show() 
        #
        plt.subplot(5, 1, 1)
        plt.plot(t, self.state[t,0])
        plt.plot(t, self.state_hat[t,0])
        plt.xlabel(' ')
        plt.ylabel('unit ')
        plt.title(' ')
        plt.legend(['position'])
        
        plt.subplot(5, 1, 2)
        plt.plot(t, self.state[t,1])
        plt.plot(t, self.state_hat[t,1])
        plt
        plt.show()  # You must call plt.show() to make graphics appear.

import math
import numpy as np
import matplotlib.pyplot as plt

def main():
    parameterCartPole=(1.0, 0.1, 0.5) #( massCart, massPole, length)
    plant=CartPoleV0(parameterCartPole)
    
    nnStructure=(5,30,15,4,)
    timeStepRange=20
    neuronNet=nnLearner1(nnStructure,timeStepRange)
    task=taskLocal1(timeStepRange)
    
    episodeRange=5
    for i_episode in range(episodeRange):
        task.reset(plant)
        task.genAction(i_episode,plant)
        task.step(plant)
        task.evaluation(i_episode,neuronNet)
    task.plotFig()
        
if __name__ == '__main__':  
        main()        
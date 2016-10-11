# -*- coding: utf-8 -*-
"""
Created on Sun Oct 02 12:48:12 2016

@author: a8zn5
"""
import numpy as np
import matplotlib.pyplot as plt

class taskLocal1(object):
    
    def __init__(self,timeStepRange):
        self.timeStepRange=timeStepRange
        self.state=np.zeros((timeStepRange,4))
        self.state_hat=np.zeros((timeStepRange,4))
        
#    def reset(self, plant, initOption=1):
#        plant.reset(initOption)
        
    def reset(self, plant, initOption=1): #
        if initOption == 1: # zero initial value
            self.state[0,:]=np.zeros((1,4))
        elif initOption==2: # random initial value
            self.state[0,:]=0.2*np.random.random((1,4))-0.1
        else: # pointed initial value
            pass # keep self.state in __init__
                
    def genAction(self, plant, actionOption=1 ):
        if actionOption == 1: # random
            self.action=10*np.random.random((self.timeStepRange,1)) - 5  
        elif actionOption == 2: # a step input
            self.action=np.ones((self.timeStepRange,1))*plant.force_mag
            self.action[0,0]= 0
        else: pass # other action plan
        
    def stepWise(self,plant):
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
        plt.xlabel(' time-step ')
        plt.ylabel('N')
        plt.title(' Force applied on converted Pendulum ')
        plt.legend(['force'])   
        plt.show() 
        #
#        plt.subplot(4, 1, 1)
        plt.plot(t, self.state[t,0])
        plt.plot(t, self.state_hat[t,0])
        plt.xlabel('time-step ')
        plt.ylabel('m ')
        plt.title(' cart position ')
        plt.legend(['position','prediction'])
        plt.show()
        
#        plt.subplot(4, 1, 2)
        plt.plot(t, self.state[t,1])
        plt.plot(t, self.state_hat[t,1])
        plt.xlabel('time-step ')
        plt.ylabel('m/s ')
        plt.title('cart speed ')
        plt.legend(['speed','prediction'])
        plt.show()
        
#        plt.subplot(4, 1, 3)
        plt.plot(t, self.state[t,2])
        plt.plot(t, self.state_hat[t,2])
        plt.xlabel('time-step ')
        plt.ylabel('radian ')
        plt.title('angle of converted pendulum ')
        plt.legend(['angle','prediction'])
        plt.show()
                
#        plt.subplot(4, 1, 4)
        plt.plot(t, self.state[t,3])
        plt.plot(t, self.state_hat[t,3])
        plt.xlabel('time-step ')
        plt.ylabel('radian/s ')
        plt.title('angle speed ')
        plt.legend(['angle speed','prediction'])
        plt.show()  # You must call plt.show() to make graphics appear.
        
 
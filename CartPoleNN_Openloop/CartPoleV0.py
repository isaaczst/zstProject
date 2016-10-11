# -*- coding: utf-8 -*-
"""
Created on Sun Oct 02 13:06:20 2016

@author: a8zn5
"""
import math
import numpy as np
class CartPoleV0(object):
    def __init__(self, Parameter):
        self.gravity = 9.8
        self.masscart = Parameter[0]
        self.masspole = Parameter[1]
        self.total_mass = (self.masspole + self.masscart)
        self.length = Parameter[2] # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10
        self.tau = 0.02  # seconds between state updates
        self.theta_threshold_radians = Parameter[3] * 2 * math.pi / 360
        self.x_threshold = Parameter[4]     #unit of position, length
        self.state = np.array([0,0,0,0]) # in case of ...

#    def reset(self, initOption=1): #
#        if initOption == 1: # zero initial value
#            self.state=np.zeros((1,4))
#        elif initOption==2: # random initial value
#            self.state=0.2*np.random.random((1,4))-0.1
#        else: # pointed initial value
#            pass # keep self.state in __init__
            
    
    def step(self, state, action):

        x, x_dot, theta, theta_dot = state
        force = action[0]
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
#        done =  x < -self.x_threshold \
#                or x > self.x_threshold \
#                or theta < -self.theta_threshold_radians \
#                or theta > self.theta_threshold_radians
#        done = bool(done)
   
        return np.array(state)  #, reward, done, {}
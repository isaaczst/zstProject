# -*- coding: utf-8 -*-
"""
Created on Sun Oct 02 12:45:42 2016

@author: a8zn5

an easy version of NN
-simple function fitting task

"""

from CartPoleV0 import CartPoleV0
from TaskLocal import taskLocal1 #import *
from NNLearner import nnLearner1

#def main():
ParameterCartPole=(1.0, 0.1, 0.5, 12, 2.4) #( massCart, massPole, length, critics_radian, critics_unit_position )
plant=CartPoleV0(ParameterCartPole)

hyperParameter=(5,30,15,4,)
timeStepRange=20
neuronNet=nnLearner1(hyperParameter,timeStepRange)
task=taskLocal1(timeStepRange)

episodeRange=10
for i_episode in range(episodeRange):
    task.reset(plant,1)
    task.genAction(plant,1)
    task.stepWise(plant)
    task.evaluation(i_episode,neuronNet)
task.plotFig()
#
task.reset(plant,1)
task.genAction(plant,2)
task.stepWise(plant)
task.evaluation(100,neuronNet)
task.plotFig()       
#if __name__ == '__main__':  
#        main()

    



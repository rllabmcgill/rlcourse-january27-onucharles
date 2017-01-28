import numpy as np

class Bandit(object):

    def __init__(self, no_of_arms):
        self.no_of_arms = no_of_arms
        self.action_values = self.initialiseArmValues()

    def initialiseArmValues(self):
        #set action values from gaussian of mean 0 and std 1.
        action_values = np.random.normal(0, 1, self.no_of_arms)
        return action_values

    def getReward(self, action, timestep):
        #get value of requested action
        action_mean = self.action_values[action]

        #obtain reward from normal distribution centered at value.
        reward = np.random.normal(action_mean, 1, 1)
        return reward
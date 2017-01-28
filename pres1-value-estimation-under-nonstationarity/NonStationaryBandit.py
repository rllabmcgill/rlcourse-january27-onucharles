import numpy as np

class NonStationaryBandit(object):

    def __init__(self, no_of_arms, interval):
        self.no_of_arms = no_of_arms
        self.interval = interval
        self.initialiseArmValues()

    def initialiseArmValues(self):
        #set action values start off at same value.
        self.action_values = np.zeros((self.no_of_arms, 1))

    def updateArmValues(self):
        #set action values from gaussian of mean 0 and std 1.
        self.action_values = np.random.normal(0, 1, self.no_of_arms)

    def getReward(self, action, timestep):
        #reset arm values after every 'interval' timesteps
        if timestep % self.interval == 0:
            self.updateArmValues()

        # get value of requested action
        action_mean = self.action_values[action]

        # obtain reward from normal distribution centered at value.
        reward = np.random.normal(action_mean, 1, 1)
        return reward
from NonStationaryBandit import NonStationaryBandit
from Agent import Agent
import numpy as np
import matplotlib.pyplot as plt
import pickle

#initialise variables
no_of_iterations = 1000
no_of_time_steps = 1000
interval_of_nonstat = 200

#method 1: sample-average value estimates.
sample_avg_rewards = np.zeros((no_of_time_steps, no_of_iterations))
for i in np.arange(no_of_iterations):
    bandit = NonStationaryBandit(10, interval_of_nonstat)
    agent = Agent(bandit, no_of_time_steps, 0.1, 0)
    agent.learn()
    sample_avg_rewards[:, i] = agent.rewards_per_time
    print('1. iteration ' + str(i))

mean_sample_avg_rewards = np.mean(sample_avg_rewards, axis=1)

#method 2: constant step-size parameter
const_step_rewards = np.zeros((no_of_time_steps, no_of_iterations))
for i in np.arange(no_of_iterations):
    bandit = NonStationaryBandit(10, interval_of_nonstat)
    #bandit = Bandit(10)
    agent = Agent(bandit, no_of_time_steps, 0.1, 1)
    agent.learn()
    const_step_rewards[:, i] = agent.rewards_per_time
    print('2. iteration ' + str(i))

mean_const_step_rewards = np.mean(const_step_rewards, axis=1)

all_rewards = np.zeros((no_of_time_steps, 2))
all_rewards[:,0] = mean_sample_avg_rewards
all_rewards[:,1] = mean_const_step_rewards


plt.plot(np.arange(no_of_time_steps) + 1, mean_sample_avg_rewards, label='sample-averaging')
plt.plot(np.arange(no_of_time_steps) + 1, mean_const_step_rewards, label='constant step-size')
plt.legend(loc='upper right')
plt.show()

#save to file
pickle.dump(all_rewards, open("nonstationary_all_rewards.pkl", "wb"))
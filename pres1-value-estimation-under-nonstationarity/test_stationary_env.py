from Bandit import Bandit
from Agent import Agent
import numpy as np
import matplotlib.pyplot as plt
import pickle

#initialise variables
no_of_iterations = 2000
no_of_time_steps = 1000
all_rewards = np.zeros((no_of_time_steps, no_of_iterations))

#learn
for i in np.arange(no_of_iterations):
    bandit = Bandit(10)
    agent = Agent(bandit, no_of_time_steps, 0.1, 0)
    agent.learn()
    all_rewards[:, i] = agent.rewards_per_time
    # print(bandit.action_values)
    # print(agent.action_selection_counts)
    # print(agent.action_value_estimate)

#plot
plt.plot(np.arange(0,1000) + 1, np.mean(all_rewards, axis=1))
plt.show()

#save to file
pickle.dump(all_rewards, open("stationary_rewards_eps0.1.pkl", "wb"))
#all_rewards = pickle.load(open("save.pkl", "rb"))
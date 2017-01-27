import pickle
import numpy as np
import matplotlib.pyplot as plt

#first experiment
all_rewards = pickle.load(open('saved/stationary_rewards_eps0.pkl', 'rb'))
plt.plot(np.arange(0,1000) + 1, np.mean(all_rewards, axis=1), label='eps = 0')

all_rewards = pickle.load(open('saved/stationary_rewards_eps0.01.pkl', 'rb'))
plt.plot(np.arange(0,1000) + 1, np.mean(all_rewards, axis=1), label='eps = 0.01')

all_rewards = pickle.load(open('saved/stationary_rewards_eps0.1.pkl', 'rb'))
plt.plot(np.arange(0,1000) + 1, np.mean(all_rewards, axis=1), label='eps = 0.1')

plt.xlabel('Steps', fontsize=14)
plt.ylabel('Average reward', fontsize=14)
plt.legend(loc='bottom right')
plt.show()

#second experiment
# all_rewards = pickle.load(open('saved/nonstationary_all_rewards.pkl', 'rb'))
#
# no_of_time_steps = all_rewards.shape[0]
# mean_sample_avg_rewards = all_rewards[:, 0]
# mean_const_step_rewards = all_rewards[:, 1]
#
# plt.plot(np.arange(no_of_time_steps) + 1, mean_sample_avg_rewards, label='sample-averaging')
# plt.plot(np.arange(no_of_time_steps) + 1, mean_const_step_rewards, label='constant step-size')
# plt.xlabel('Steps', fontsize=14)
# plt.ylabel('Average reward', fontsize=14)
# plt.legend(loc='upper right')
# plt.show()
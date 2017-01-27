import numpy as np

class Agent(object):

    def __init__(self, bandit, steps, epsilon, value_estimate_method):
        self.bandit = bandit
        self.no_of_steps = steps
        self.epsilon = epsilon
        self.action_value_estimate = np.zeros(bandit.no_of_arms)
        self.action_selection_counts = np.ones(bandit.no_of_arms)
        self.rewards_per_time = np.zeros(steps)
        if value_estimate_method == 0:
            self.value_function = lambda reward, cur_value, n: cur_value + (1/n) * (reward - cur_value)
        else:
            self.value_function = lambda reward, cur_value, n: cur_value + 0.1 * (reward - cur_value)

    def learn(self):
        for i in np.arange(self.no_of_steps):
            action = self.selectAction()
            reward = self.bandit.getReward(action, i + 1)
            cur_value = self.action_value_estimate[action]
            cur_selection_count = self.action_selection_counts[action]
            #new_value = cur_value + (1/cur_selection_count) * (reward - cur_value)
            new_value = self.value_function(reward, cur_value, cur_selection_count)

            self.rewards_per_time[i] = reward
            self.action_selection_counts[action] += 1
            self.action_value_estimate[action] = new_value

    def selectAction(self):
        #0 for explore. 1 for exploit.
        choice = np.random.choice([0, 1], 1, p=[self.epsilon, 1 - self.epsilon])

        if (choice == 0):
            #choose random action (uniformly)
            action = np.random.randint(0, self.bandit.no_of_arms)
        elif (choice == 1):
            #exploit: choose action with max value
            action = np.argmax(self.action_value_estimate)

        return action
"""
2019/01/27

Hands on Machine Learning with scikit-learn and TensorFlow  -- Chapter 16

Example 3. Markov Decision Process.
"""

import numpy as np


transition_probabilities = np.array([
    [[0.7, 0.3, 0.0], [1.0, 0.0, 0.0], [0.8, 0.2, 0.0]],    # in s0, if action a0 then proba 0.7 to state s0 and 0.3 to state s1, etc.
    [[0.0, 1.0, 0.0], None, [0.0, 0.0, 1.0]],
    [None, [0.8, 0.1, 0.1], None],
])

rewards = np.array([
    [[10, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [40, 0, 0], [0, 0, 0]],
])

possible_actions = [[0, 1, 2], [0, 2], [1]]


def policy_fire(_state):
    return [0, 2, 1][_state]


def policy_random(_state):
    return np.random.choice(possible_actions[_state])


def policy_safe(_state):
    return [0, 0, 1][_state]


class MDPEnvironment(object):
    def __init__(self, start_state=0):
        self.start_state = start_state
        self.total_rewards = None
        self.state = None
        self.reset()

    def reset(self):
        self.total_rewards = 0
        self.state = self.start_state

    def step(self, _action):
        n_state = np.random.choice(range(3), p=transition_probabilities[self.state][_action])
        _reward = rewards[self.state][_action][n_state]
        self.state = n_state
        self.total_rewards += _reward
        return self.state, _reward


def run_episode(_policy, n_steps, start_state=0, display=True):
    env = MDPEnvironment()
    if display:
        print('State (+Rewards): ', end='\t')
    for step in range(n_steps):
        if display:
            if step == 10:
                print('...', end='\t')
            elif step < 10:
                print(env.state, end='\t')
        action = _policy(env.state)
        state, reward = env.step(action)
        if display and step < 10:
            if reward:
                print('({})'.format(reward), end='\t')
    if display:
        print('Total rewards = ', env.total_rewards)
    return env.total_rewards


for policy in (policy_fire, policy_random, policy_safe):
    all_totals = []
    print(policy.__name__)
    for episode in range(1000):
        all_totals.append(run_episode(policy, n_steps=100, display=(episode < 5)))
    print('Summary: mean={:.1f}, std={:1f}, min={}, max={}'.format(np.mean(all_totals), np.std(all_totals),
                                                                   np.min(all_totals), np.max(all_totals)))
    print()


# Run the Q-Value Iteration algorithm
print('\nRunning Q-Value Iteration Algorithm')
gamma = 0.95    # The discount factor
n_iterations = 100
n_steps = 20
alpha = 0.01
exploration_policy = policy_random

q_values = np.full((3, 3), -np.inf)        # -inf for impossible actions
for state, action in enumerate(possible_actions):
    q_values[state, action] = 0.0         # Initial value = 0.0 for all possible action

print('Initial q_value:\n', q_values)
env = MDPEnvironment()

for step in range(n_steps):
    action = exploration_policy(env.state)
    state = env.state
    next_state, reward = env.step(action)
    next_value = np.max(q_values[next_state])   # Greedy policy
    print('state: {}\taction:{}\tnext state: {}\treward: {}\tnext_value: {}'.format(state, action,
                                                                                    next_state, reward, next_value))
    q_values[state, action] = (1 - alpha) * q_values[state, action] + alpha * (reward + gamma * next_value)

print('\nAfter Q-Value algorithm q_value:\n', q_values)

print('Best action: ', np.argmax(q_values, axis=1))


# Running Q-Learning Algorithm.
learning_rate0 = 0.05
learning_rate_decay = 0.1
n_iterations = 20000

s = 0       # Start from state 0

Q = np.full((3, 3), -np.inf)    # -inf for impossible actions
for state, action in enumerate(possible_actions):
    Q[state, action] = 0.0      # Initial value = 0.0, for all possible actions

for ite in range(n_iterations):
    a = np.random.choice(possible_actions[s])   # Choose an action (randomly)
    sp = np.random.choice(range(3), p=T[s, a])
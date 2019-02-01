"""
2019/01/26

Hands on Machine Learning with scikit-learn and TensorFlow  -- Chapter 16

Example 2. Policy Gradients
"""

import gym
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

env = gym.make('CartPole-v0')

# 1. Specify the network architecture
n_inputs = 4            # == env.observation_space.shape[0]
n_hidden = 4            # It's a simple task
n_outputs = 1           # Only outputs the probability of accelerating left, just one action
learning_rate = 0.01
initializer = tf.variance_scaling_initializer()

# 2. Build the neural network.
X = tf.placeholder(tf.float32, shape=[None, n_inputs])      # shape=(?, 4)
hidden = fully_connected(X, n_hidden, activation_fn=tf.nn.relu, weights_initializer=initializer)    # shape=(?, 4)
logits = fully_connected(hidden, n_outputs, activation_fn=None, weights_initializer=initializer)    # shape=(?, 1)
outputs = tf.nn.sigmoid(logits)     # shape=(?, 1)

# 3. Select a random action based on the estimated probabilities.
p_left_and_right = tf.concat(values=[outputs, 1 - outputs], axis=1)     # shape=(?, 2)
action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)        # shape=(?, 1)


y = 1. - tf.to_float(action)        # shape=(?, 1)
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(cross_entropy)
gradients = [grad for grad, variable in grads_and_vars]

# print(len(grads_and_vars), grads_and_vars)
# print(len(gradients), gradients)

gradient_placeholders = []
grads_and_vars_feed = []
for grad, variable in grads_and_vars:
    _gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
    gradient_placeholders.append(_gradient_placeholder)
    grads_and_vars_feed.append((_gradient_placeholder, variable))
# print('gradient_placeholders: ', gradient_placeholders)
# print('grads_and_vars_feed: ', grads_and_vars_feed)
training_op = optimizer.apply_gradients(grads_and_vars_feed)

init = tf.global_variables_initializer()
saver = tf.train.Saver()


def discount_rewards(rewards, _discount_rate):
    discounted_rewards = np.empty(len(rewards))
    cumulative_rewards = 0
    for _step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[_step] + cumulative_rewards * _discount_rate
        discounted_rewards[_step] = cumulative_rewards
    return discounted_rewards


def discount_and_normalize_rewards(_all_rewards, _discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, _discount_rate) for rewards in _all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discount_reward - reward_mean) / reward_std for discount_reward in all_discounted_rewards]


def _max_len(_rewards):
    max_len = 0
    for rew in _rewards:
        max_len = max(max_len, len(rew))
    return max_len


# print(discount_rewards([10, 0, -50], discount_rate=0.8))
# print(discount_and_normalize_rewards([[10, 0, -50], [10, 20]], discount_rate=0.8))


n_iterations = 250          # number of training iterations
n_max_steps = 1000          # max step per episode
n_games_per_update = 10     # train the policy every 10 episodes
save_iterations = 10        # save the model every 10 training iterations
discount_rate = 0.95

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):   # 250
        print("\rIteration: {}".format(iteration), end="\t")
        all_rewards = []    # All sequences of raw rewards for each episode
        all_gradients = []  # Gradients saved at each step of each episode
        for game in range(n_games_per_update):  # 10
            current_rewards = []    # All raw rewards from the current episode
            current_gradients = []  # All gradients from the current episode
            obs = env.reset()
            for step in range(n_max_steps):     # 1000
                action_val, gradients_val = sess.run([action, gradients],   # action = shape=(?, 1)
                                                     feed_dict={X: obs.reshape(1, n_inputs)})   # one obs
                obs, reward, done, info = env.step(action_val[0][0])
                current_rewards.append(reward)
                current_gradients.append(gradients_val)
                if done:
                    break
            all_rewards.append(current_rewards)
            all_gradients.append(current_gradients)

        print('All_Rewards: ', len(all_rewards), _max_len(all_rewards), all_rewards)
        # At this point we have run the policy for 10 episodes, and we are ready
        # for a policy update using the algorithm described earlier.
        all_rewards = discount_and_normalize_rewards(all_rewards, discount_rate)
        feed_dict = {}
        for var_index, gradient_placeholder in enumerate(gradient_placeholders):
            # multiply the gradients by the action scores, and compute the mean
            mean_gradients = np.mean([reward * all_gradients[game_index][step][var_index]
                                      for game_index, rewards in enumerate(all_rewards)
                                      for step, reward in enumerate(rewards)], axis=0)
            feed_dict[gradient_placeholder] = mean_gradients
        sess.run(training_op, feed_dict=feed_dict)
        if iteration % save_iterations == 0:
            saver.save(sess, './my_policy_net_pg.ckpt')

"""
2019/01/26

Hands on Machine Learning with scikit-learn and TensorFlow  -- Chapter 16

Example 1. OpenAI
"""

import gym
import numpy as np


env = gym.make('CartPole-v0')
print('env: ', env)

obs = env.reset()
print('obs: ', obs)

img = env.render(mode='rgb_array')
print('Shape of img: ', img.shape)

print('Posible actions: ', env.action_space, env.action_space.shape)
print('Observation space: ', env.observation_space, env.observation_space.shape[0])

action = 1  # accelerate right
obs, reward, done, info = env.step(action)
print('Outputs of env.step(action): ', obs, reward, done, info)


# A simple policy that accelerates left when the pole is leaning toward the left
# and accelerates right when the pole is leaning toward the right.
def basic_policy(_obs):
    angle = _obs[2]
    return 0 if angle < 0 else 1


total_rewards = []
for episode in range(500):
    episode_rewards = 0
    obs = env.reset()
    for step in range(1000):
        action = basic_policy(obs)
        obs, reward, done, info = env.step(action)
        episode_rewards += reward
        if done:
            break
    total_rewards.append(episode_rewards)

print('total_rewards: ', total_rewards)
print('Mean: ', np.mean(total_rewards))
print('Std: ', np.mean(total_rewards))
print('Max: ', np.max(total_rewards))
print('Min: ', np.min(total_rewards))

env.close()

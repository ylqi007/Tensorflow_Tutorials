"""
2019/01/28

Hands on Machine Learning with scikit-learn and TensorFlow  -- Chapter 16

Example 4. Play Ms. Pac-Man Using the DQN Algorithm
"""

import os
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import matplotlib.animation as animation

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


env = gym.make('MsPacman-v0')
obs = env.reset()
print('shape of obs: ', obs.shape)      # (210, 160, 3)
print('Action space: ', env.action_space, env.action_space.n)   # Discrete(9) 9


def plot_environment(env, figsize=(5, 4)):
    plt.close()
    plt.figure(figsize=figsize)
    img = env.render(mode='rgb_array')
    plt.imshow(img)
    plt.axis('off')
    plt.show()


# Plot the initialize the environment.
# plot_environment(env)

for step in range(110):
    env.step(3)     # left
for step in range(40):
    env.step(8)     # lower-left

# Plot the environment after some steps.
# plot_environment(env)

# The step() function returns several important object:
obs, reward, done, info = env.step(0)

print('After step(0): ', type(obs), obs.shape, reward, done, info)

# Play one full game, by moving in random directions for 10 steps at a time, recording each frame.
frames = []
n_max_steps = 1000
n_change_steps = 10

obs = env.reset()
for step in range(n_max_steps):
    img = env.render(mode='rgb_array')
    frames.append(img)
    if step % n_change_steps == 0:
        _action = env.action_space.sample()      # Play randomly
    obs, reward, done, info = env.step(_action)
    if done:
        break


# Show the animation
def update_scene(num, _frames, patch):
    patch.set_data(_frames[num])
    return patch


def plot_animation(_frames, repeat=False, interval=40):
    plt.close()     # or else nbagg sometimes plots in the previous cell
    fig = plt.figure()
    patch = plt.imshow(_frames[0])
    plt.axis('off')
    return animation.FuncAnimation(fig, update_scene, fargs=(_frames, patch), frames=len(frames),
                                   repeat=repeat, interval=interval)


video = plot_animation(frames)
plt.show()

env.close()


# Creating the MsPacman envirionment
env = gym.make('MsPacman-v0')
obs = env.reset()

mspacman_color = 210 + 164 + 74


def preprocess_observation(_obs):
    _img = _obs[1:176:2, ::2]       # Crop and downsize
    _img = _img.sum(axis=2)         # To greyscale, i.e. R + G + B
    _img[_img == mspacman_color] = 0  # Improve contrast  ?
    _img = (_img // 3 - 128).astype(np.int8)    # Normalize from -128 to 127
    # print(type(_img), _img.shape)
    return _img.reshape(88, 80, 1)


img = preprocess_observation(obs)
print(img.shape)


# Build Deep Q-Learning Network
reset_graph()

input_height = 88
input_width = 80
input_channels = 1
conv_n_maps = [32, 64, 64]
conv_kernel_sizes = [(8, 8), (4, 4), (3, 3)]
conv_strides = [4, 2, 1]
conv_paddings = ["SAME"] * 3
conv_activation = [tf.nn.relu] * 3
n_hidden_in = 64 * 11 * 10      # conv3 has 64 maps of 11*10
n_hidden = 512
hidden_activation = tf.nn.relu
n_outputs = env.action_space.n  # 9 discrete actions are available

initializer = tf.variance_scaling_initializer()


def q_network(x_state, name):
    prev_layer = x_state
    with tf.variable_scope(name) as scope:
        # Conv layers
        for n_maps, kerner_size, strides, padding, activation in zip(
                conv_n_maps, conv_kernel_sizes, conv_strides, conv_paddings, conv_activation):
            prev_layer = tf.layers.conv2d(prev_layer, filters=n_maps, kernel_size=kerner_size, strides=strides,
                                          padding=padding, activation=activation, kernel_initializer=initializer)
        # Hidden layers
        last_conv_layer_flat = tf.reshape(prev_layer, shape=[-1, n_hidden_in])
        hidden = tf.layers.dense(last_conv_layer_flat, n_hidden_in,
                                 activation=hidden_activation, kernel_initializer=initializer)
        # Fully output layer.
        outputs = tf.layers.dense(hidden, n_outputs, kernel_initializer=initializer)
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
    # print('\tTrainable_vars: ', trainable_vars)
    trainable_vars_by_name = {var.name[len(scope.name):]: var for var in trainable_vars}
    # print('\tTraniable_vars_by_name: ', trainable_vars_by_name)
    return outputs, trainable_vars_by_name


X_state = tf.placeholder(tf.float32, shape=[None, input_height, input_width, input_channels])

online_q_values, online_vars = q_network(X_state, name='q_networks/online')
target_q_values, target_vars = q_network(X_state, name='q_networks/target')

copy_ops = [target_var.assign(online_vars[var_name]) for var_name, target_var in target_vars.items()]
copy_online_to_target = tf.group(*copy_ops)
print('online_vars: ', online_vars)
print('online_q_value: ', online_q_values)


learning_rate = 0.001
momentum = 0.95

with tf.variable_scope('train'):
    X_action = tf.placeholder(tf.int32, shape=[None])
    y = tf.placeholder(tf.float32, shape=[None, 1])
    print('\tTimes: ', online_q_values * tf.one_hot(X_action, n_outputs))
    q_value = tf.reduce_sum(online_q_values * tf.one_hot(X_action, n_outputs), axis=1, keepdims=True)
    print('q_value: ', q_value)

    error = tf.abs(y - q_value)
    clipped_error = tf.clip_by_value(error, 0.0, 1.0)
    linear_error = 2 * (error - clipped_error)
    loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)

    global_step = tf.Variable(0, trainable=False, name='global_step')
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
    training_op = optimizer.minimize(loss, global_step=global_step)

print('tf.one_hot(X_action, n_outputs): ', tf.one_hot(X_action, n_outputs))
print('# error: ', error)
print('# clipped_error: ', clipped_error)
print('# linear_error: ', linear_error)
print('# loss: ', loss)
print('## global_step: ', global_step)


init = tf.global_variables_initializer()
saver = tf.train.Saver()


class ReplayMemory:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.buf = np.empty(shape=maxlen, dtype=np.object)
        self.index = 0
        self.length = 0

    def append(self, data):
        self.buf[self.index] = data
        self.length = min(self.length + 1, self.maxlen)
        self.index = (self.index + 1) % self.maxlen

    def sample(self, batch_size, with_replacement=True):
        if with_replacement:
            indices = np.random.randint(self.length, size=batch_size)
        else:
            indices = np.random.permutation(self.length)[:batch_size]
        return self.buf[indices]


replay_memory_size = 500000
replay_memory = ReplayMemory(replay_memory_size)


def sample_memories(batch_size):
    cols = [[], [], [], [], []]     # state, action, reward, next_state, continue
    for memory in replay_memory.sample(batch_size):
        for col, value in zip(cols, memory):
            col.append(value)
    cols = [np.array(col) for col in cols]
    return cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1)


# epsilon-greedy policy, and gradually decrease epsilong from 1.0 to 0.1
eps_min = 0.1
eps_max = 1.0
eps_decay_steps = 2000000


def epsilon_greedy(q_values, step):
    epsilon = max(eps_min, eps_max - (eps_max - eps_min) * step / eps_decay_steps)
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs)     # random action
    else:
        return np.argmax(q_values)              # optimal action


n_steps = 4000000       # total number of training steps
training_start = 1000   # Start training after 10000 game iterations
training_interval = 4   # Run a training step every 4 game iterations
save_steps = 1000       # Save the model every 1000 training steps
copy_steps = 10000      # Copy online DQN to target DQN every 10000 training steps
discount_rate = 0.99    # i.e. gamma
skip_start = 90         # Skip the start of every game (it's just waiting time).
batch_size = 50
iteration = 0           # Game iterations
checkpoint_path = './my_dqn.ckpt'
done = True             # env needs to be reser

# A few variables for tracking progress.
loss_val = np.infty
game_length = 0
total_max_q = 0
mean_max_q = 0.0


with tf.Session() as sess:
    if os.path.isfile(checkpoint_path + '.index'):
        saver.restore(sess, checkpoint_path)
    else:
        init.run()
        copy_online_to_target.run()

    while True:
        step = global_step.eval()
        if step >= n_steps:
            break
        iteration += 1
        print('\rIteration {:7}\tTraining step {}/{} ({:.1f})%\tLoss {:5f}\tMean Max-Q {:5f}'.format(
            iteration, step, n_steps, step*100 / n_steps, loss_val, mean_max_q), end='')
        if done:    # game over, start again
            obs = env.reset()
            for skip in range(skip_start):  # skip the start of each game
                obs, reward, done, info = env.step(0)
            state = preprocess_observation(obs)

        # Online DQN evaluates what to do.
        q_values = online_q_values.eval(feed_dict={X_state: [state]})
        action = epsilon_greedy(q_values, step)
        # print('\tDuring Online DQN: q_values: ', q_values)
        # print('\tDuring Online action: ', action)

        # Online DQN plays
        obs, reward, done, info = env.step(action)
        next_state = preprocess_observation(obs)

        # Memorize what happened.
        replay_memory.append((state, action, reward, next_state, 1.0 - done))
        state = next_state

        # Compute sattistics for tracking progress (not shown in the book)
        total_max_q += q_values.max()
        game_length += 1
        # print(total_max_q)
        if done:
            mean_max_q = total_max_q / game_length
            total_max_q = 0.0
            game_length = 0

        if iteration < training_start or iteration % training_interval != 0:
            continue    # Only train after warmup period and at regular intervals

        # Sample memories and use the target DQN to produce the target Q-Value.
        X_state_val, X_action_val, rewards, X_next_state_val, continues = (sample_memories(batch_size))
        # print('X_state_val: ', X_state_val)
        # print('X_action_val: ', X_action_val)
        # print('rewards: ', rewards)
        # print('X_next_state_val: ', X_next_state_val)
        # print('continue: ', continues)
        next_q_values = target_q_values.eval(feed_dict={X_state: X_next_state_val})
        max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
        y_val = rewards + continues * discount_rate * max_next_q_values
        # print('### X_state_val: ', X_state_val)
        # print('### next_q_values: ', next_q_values)

        # Train the online DQN
        _, loss_val = sess.run([training_op, loss], feed_dict={
            X_state: X_state_val, X_action: X_action_val, y: y_val})

        # Regularly copy the online DQN to the target DQN
        if step % copy_steps == 0:
            copy_online_to_target.run()

        # Save regularly
        if step % save_steps == 0:
            saver.save(sess, checkpoint_path)


with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)

    obs = env.reset()
    for step in range(n_max_steps):
        state = preprocess_observation(obs)

        # Online DQN evaluates what to do
        q_values = online_q_values.eval(feed_dict={X_state: [state]})
        action = np.argmax(q_values)

        # Online DQN plays
        obs, reward, done, info = env.step(action)

        img = env.render(mode='rgb_array')
        frames.append(img)

        if done:
            break

plot_animation(frames)

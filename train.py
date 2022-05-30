import numpy as np
import tensorflow as tf
from actor import Actor
from critic import Critic
from gym_torcs import TorcsEnv
from ou_action_noise import OUActionNoise
import matplotlib.pyplot as plt

# based on: https://keras.io/examples/rl/ddpg_pendulum/

ou_noise_steer = OUActionNoise(mean=np.zeros(1), std_deviation=float(0.3) * np.ones(1), theta=0.6)
ou_noise_speed = OUActionNoise(mean=np.full(1, 0.4), std_deviation=float(0.1) * np.ones(1))

actor = Actor().model
critic = Critic().model

target_actor = Actor().model
target_critic = Critic().model

target_actor.set_weights(actor.get_weights())
target_critic.set_weights(critic.get_weights())

critic_lr = 0.001
actor_lr = 0.0001

critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

gamma = 0.99
tau = 0.001

num_states = 23
num_actions = 2

eps = 1
exploration = 100000.


class Buffer:
    def __init__(self, capacity=50000, batch_size=64):
        self.capacity = capacity
        self.batch_size = batch_size
        self.counter = 0
        self.num_states = num_states
        self.num_actions = num_actions

        self.state_buffer = np.zeros((self.capacity, num_states))
        self.action_buffer = np.zeros((self.capacity, num_actions))
        self.reward_buffer = np.zeros((self.capacity, 1))
        self.state__buffer = np.zeros((self.capacity, num_states))

    def store(self, state_arg, action_arg, reward_arg, state__arg):
        index = self.counter % self.capacity

        self.state_buffer[index] = state_arg
        self.action_buffer[index] = action_arg
        self.reward_buffer[index] = reward_arg
        self.state__buffer[index] = state__arg

        if self.counter > self.capacity:
            self.counter = 0
        self.counter += 1

    @tf.function
    def update(self, state_batch, action_batch, reward_batch, state__batch):
        with tf.GradientTape() as tape:
            target_actions = target_actor(state__batch, training=True)
            y = reward_batch + gamma * target_critic([state__batch, target_actions], training=True)
            critic_value = critic([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
            tf.print("critic loss:", critic_loss)

        critic_grad = tape.gradient(critic_loss, critic.trainable_variables)
        critic_optimizer.apply_gradients(zip(critic_grad, critic.trainable_variables))

        with tf.GradientTape() as tape:
            actions = actor(state_batch, training=True)
            critic_value = critic([state_batch, actions], training=True)
            actor_loss = -tf.math.reduce_mean(critic_value)
            tf.print("actor loss:", actor_loss)

        actor_grad = tape.gradient(actor_loss, actor.trainable_variables)
        actor_optimizer.apply_gradients(zip(actor_grad, actor.trainable_variables))

    def learn(self):
        record_range = min(self.counter, self.capacity)
        batch_indices = np.random.choice(record_range, self.batch_size)

        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        state__batch = tf.convert_to_tensor(self.state__buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, state__batch)


@tf.function
def update_target(target_weights, weights):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


def policy(state_policy_arg, noise_steer, noise_speed):
    sampled_actions = tf.squeeze(actor(state_policy_arg))
    noise_0 = noise_steer() * max(0, eps)
    noise_1 = noise_speed() * max(0, eps)

    sampled_actions = sampled_actions.numpy()
    sampled_actions[0] += noise_0
    sampled_actions[1] += noise_1
    # sampled_actions[1] = max(sampled_actions[1], 0.25)
    legal_action = np.clip(sampled_actions, -1.0, 1.0)

    return [np.squeeze(legal_action)]


buffer = Buffer()


def scale_input(s):
    # s[0] = s[0] #/ math.pi  # ) / (2 * math.pi)
    for p in range(1, 21):
        s[p] = (s[p]) / 100
    # s[20] = (s[20] + 1) / 2
    for p in range(21, 23):
        s[p] = (s[p]) / 100
    return s


if __name__ == "__main__":
    print('Starting the experiment.')
    env = TorcsEnv(vision=False, throttle=True, gear_change=False)

    actor_weights_path = "training_1/actor.ckpt"
    target_actor_weights_path = "training_1/target_actor.ckpt"
    critic_weights_path = "training_1/critic.ckpt"
    target_critic_weights_path = "training_1/target_critic.ckpt"

    filename = 'weights'
    actor.load_weights(filename+"/actor.ckpt")
    target_actor.load_weights(filename+"/target_actor.ckpt")
    critic.load_weights(filename+"/critic.ckpt")
    target_critic.load_weights(filename+"/target_critic.ckpt")

    score_history = []
    steps_history = []
    num_episodes = 100000
    max_steps = 100000
    best_score = -10000

    for i in range(num_episodes):
        print('... starting episode ', i, ' ...')
        done = False
        score = 0
        observation = env.reset()

        state = np.hstack((observation.angle, observation.track, observation.trackPos,
                           observation.speedX, observation.speedY))
        state = scale_input(state)

        num_steps = 0
        for j in range(max_steps):
            eps -= 1 / exploration
            tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
            action = policy(tf_state, ou_noise_steer, ou_noise_speed)[0]

            observation_, reward, done, info = env.step(action)

            state_ = np.hstack((observation_.angle, observation_.track, observation_.trackPos,
                                observation_.speedX, observation_.speedY))
            state_ = scale_input(state_)

            buffer.store(state, action, reward, state_)
            buffer.learn()

            update_target(target_actor.variables, actor.variables)
            update_target(target_critic.variables, critic.variables)

            score += reward
            state = state_

            print('... episode ', i, ', step ', j, ': reward ', reward, ' ...')

            if done:
                num_steps = j
                break

        if best_score < score:
            actor.save_weights(actor_weights_path)
            target_actor.save_weights(target_actor_weights_path)
            critic.save_weights(critic_weights_path)
            target_critic.save_weights(target_critic_weights_path)
            best_score = score

        score_history.append(score)
        steps_history.append(num_steps)
        avg_score = np.mean(score_history[-100:])
        print('Episode ', i, ': score %.2f, average score %.2f' % (score, avg_score))
        arr = np.asarray(score_history)
        np.savetxt('score_history.csv', arr, delimiter=' ')
        arr = np.asarray(steps_history)
        np.savetxt('steps_history.csv', arr, delimiter=' ')

    env.end()
    print('Experiment is finished.')


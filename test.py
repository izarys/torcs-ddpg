import numpy as np
import tensorflow as tf
from actor import Actor
from critic import Critic
from gym_torcs import TorcsEnv
from ou_action_noise import OUActionNoise
import matplotlib.pyplot as plt

ou_noise_steer = OUActionNoise(mean=np.zeros(1), std_deviation=float(0.25) * np.ones(1))
ou_noise_speed = OUActionNoise(mean=np.zeros(1), std_deviation=float(0.25) * np.ones(1))

actor = Actor().model
critic = Critic().model

target_actor = Actor().model
target_critic = Critic().model

target_actor.set_weights(actor.get_weights())
target_critic.set_weights(critic.get_weights())

critic_lr = 0.001
actor_lr = 0.0001

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

gamma = 0.99
tau = 0.001

num_states = 23
num_actions = 2


def scale_input(s):
    # s[0] = s[0] #/ math.pi  # ) / (2 * math.pi)
    for p in range(1, 21):
        s[p] = (s[p]) / 5
    # s[20] = (s[20] + 1) / 2
    for p in range(21, 23):
        s[p] = (s[p]) / 5
    return s


def policy(state_policy_arg, noise_steer, noise_speed):
    sampled_actions = tf.squeeze(actor(state_policy_arg))

    sampled_actions = sampled_actions.numpy()

    legal_action = np.clip(sampled_actions, -1.0, 1.0)

    return [np.squeeze(legal_action)]


if __name__ == "__main__":
    print('Starting the experiment.')
    env = TorcsEnv(vision=False, throttle=True, gear_change=False)

    path = 'weights'
    actor_weights_path = path+"/actor.ckpt"
    target_actor_weights_path = path+"/target_actor.ckpt"
    critic_weights_path = path+"/critic.ckpt"
    target_critic_weights_path = path+"/target_critic.ckpt"

    score_history = []
    num_episodes = 2500
    max_steps = 100000

    actor.load_weights(actor_weights_path)
    target_actor.load_weights(target_actor_weights_path)
    critic.load_weights(critic_weights_path)
    target_critic.load_weights(target_critic_weights_path)

    for i in range(num_episodes):
        print('... starting episode ', i, ' ...')
        done = False
        score = 0
        observation = env.reset()

        state = np.hstack((observation.angle, observation.track, observation.trackPos,
                           observation.speedX, observation.speedY))
        state = scale_input(state)

        for j in range(max_steps):
            tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
            action = policy(tf_state, ou_noise_steer, ou_noise_speed)[0]

            observation_, reward, done, info = env.step(action)

            state_ = np.hstack((observation_.angle, observation_.track, observation_.trackPos,
                                observation_.speedX, observation_.speedY))
            state_ = scale_input(state_)

            score += reward
            state = state_

            print('... episode ', i, ', step ', j, ': reward ', reward, ' ...')

            if done:
                break

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        print('Episode ', i, ': score %.2f, average score %.2f' % (score, avg_score))
        arr = np.asarray(score_history)
        np.savetxt('history.csv', arr, delimiter=' ')

    env.end()
    print('Experiment is finished.')

    plt.plot(score_history)
    plt.savefig('plot.png')
    plt.show()




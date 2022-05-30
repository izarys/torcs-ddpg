from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Input
import tensorflow as tf


class Actor(object):
    def __init__(self, num_states=23, num_actions=2, layer1_size=200, layer2_size=100,
                 batch_size=32, tau=0.001, learning_rate=0.0001):
        self.num_states = num_states
        self.num_actions = num_actions
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.batch_size = batch_size
        self.tau = tau
        self.learning_rate = learning_rate

        self.model = self.build_network()

    def build_network(self):
        print("... building actor network ...")
        input_values = Input(shape=(self.num_states,))
        speed_init = tf.random_uniform_initializer(minval=0, maxval=1)
        steer_init = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)

        layer1 = Dense(self.layer1_size, activation='relu')(input_values)
        layer2 = Dense(self.layer2_size, activation='relu')(layer1)

        steering = Dense(1, activation='tanh', kernel_initializer=speed_init)(layer2)
        speed = Dense(1, activation='tanh', kernel_initializer=steer_init)(layer2)
        outputs = keras.layers.Concatenate()([steering, speed])

        model = Model(inputs=input_values, outputs=outputs)
        return model





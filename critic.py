from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Input


class Critic(object):
    def __init__(self, num_states=23, num_actions=2, layer1_size=200, layer2_size=100):
        self.num_states = num_states
        self.num_actions = num_actions
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size

        self.model = self.build_network()

    def build_network(self):
        print('... building critic network ...')
        state_values = Input(shape=[self.num_states])
        action_values = Input(shape=[self.num_actions])

        layer1 = Dense(self.layer1_size, activation='relu')(state_values)
        layer2 = Dense(self.layer2_size, activation='linear')(action_values)
        layer3 = Dense(self.layer2_size, activation='linear')(layer1)

        concat = keras.layers.Concatenate()([layer2, layer3])
        layer4 = Dense(self.layer2_size, activation='relu')(concat)
        outputs = Dense(self.num_actions, activation='linear')(layer4)

        model = Model(inputs=[state_values, action_values], outputs=outputs)

        return model

from random import randint

import gym
import gym_adversarial
import numpy as np
import tensorflow as tf
from keras import layers, Sequential
import keras
from keras.layers import Permute

from consts import ACTIONS
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


def buid_model(window_length: int, nb_actions: int):
    activation = 'relu'
    img_rows, img_cols, img_colors = 28, 28, 1

    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=(window_length, img_rows, img_cols)))
    model.add(layers.Conv2D(8, kernel_size=(3, 3), activation=activation))
    model.add(layers.Conv2D(8, kernel_size=(3, 3), activation=activation))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation=activation))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(nb_actions))
    model.add(layers.Activation('softmax', name='y_pred')) # linear
    return model

class AdversarialMNISTProcessor(Processor):
    def process_observation(self, observation):
        image = observation.image.reshape(28,28)
        return image



def main():
    WINDOW_LENGTH = 4

    env = gym.make('adver-v0')
    nb_actions = env.action_space.n
    rl_model = buid_model(WINDOW_LENGTH, nb_actions)
    processor = AdversarialMNISTProcessor()
    # rl_model.compile(loss=keras.losses.categorical_crossentropy,
    #                     optimizer=keras.optimizers.Adadelta(),
    #                     metrics=[keras.metrics.CategoricalAccuracy()])
    print("Model was compiled")
    print(rl_model.output)

    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)

    dqn = DQNAgent(rl_model, nb_actions=nb_actions, memory=memory, processor=processor,)
    dqn.compile(Adam(lr=.00025), metrics=['mae'])
    dqn.fit(env, nb_steps=1750000)

    env.reset()  # Creates an image pair from a random mnist digit



    for i in range(20):
        action = randint(0, len(ACTIONS)-1)
        obs, reward, done, _ = env.step(action) # The agent takes a step. Valid steps fall within [0,24]
        print (f"reward={reward:.2f}")
        env.render() # Shows the current state of the environment
    # env.reset() # Creates an image pair from a random mnist digit
    # env.render() # Shows the current state of the environment
if __name__ == '__main__':
    main()

from random import randint

import gym
import gym_adversarial
import numpy as np
import tensorflow as tf
from keras import layers, Sequential
import keras
from keras.layers import Permute

from gym_adversarial.envs.consts import ACTIONS, WINDOW_LENGTH, DQN_WEIGHTS_FILENAME, DQN_LOG_FILENAME, NB_ACTIONS, \
    MEMORY_LIMIT, TARGET_MODEL_UPDATE, WEIGHTS_CHECKPOINT, LOG_CHECKPOINT, DQN_LR
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
import click



def buid_model(window_length: int, nb_actions: int):
    activation = 'relu'
    img_rows, img_cols, img_colors = 28, 28, 1

    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=(window_length, img_rows, img_cols)))
    model.add(layers.Conv2D(8, kernel_size=(3, 3), activation=activation))
    model.add(layers.Conv2D(8, kernel_size=(3, 3), activation=activation))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.1))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation=activation))
    model.add(layers.Dense(64, activation=activation))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(nb_actions))
    model.add(layers.Activation('linear', name='q_pred'))
    # Because in the final layer you are approximating a real state action value for each action. You output to a linear layer as your Q value estimate can generally take on any real value. And then you add a mean squares error loss with the linear layer output.
    #
    # This set up is similar any general regression problem with a neural nnetwork.
    #
    # Relu is used quite often but for hidden layer activation.
    #
    # Softmax output is quite popular for representing the policy network/function for discrete actions. But for Q learning, we are learning an approximate estimate of the Q value for each state action pair, which is a regression problem.
    return model


class AdversarialMNISTProcessor(Processor):
    def process_observation(self, observation):
        image = observation.image.reshape(28, 28)
        return image

def init_dqn(nb_steps : int):
    rl_model = buid_model(WINDOW_LENGTH, NB_ACTIONS)
    processor = AdversarialMNISTProcessor()
    memory = SequentialMemory(limit=MEMORY_LIMIT, window_length=WINDOW_LENGTH)
    # Select a policy. We use eps-greedy action selection, which means that a random action is selected with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that the agent initially explores the environment (high eps) and then gradually sticks to what it knows (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05 so that the agent still performs some random actions. This ensures that the agent cannot get stuck.

    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1,
                                  value_test=.05, nb_steps=nb_steps)

    dqn = DQNAgent(model=rl_model, nb_actions=NB_ACTIONS, policy=policy, memory=memory,
                   processor=processor, target_model_update=TARGET_MODEL_UPDATE,
                   enable_double_dqn=True,
                   train_interval=4, delta_clip=1.)
    dqn.compile(Adam(lr=DQN_LR), metrics=['mae'])
    return dqn

def start_policy(obs):
    return ACTIONS.index("CLOSET_CLUSTER")

@click.command()
@click.argument('fit_steps', type=int)
@click.option('--cont', default=False)
def fit_dqn(fit_steps: int, cont: bool):
    training_env = gym.make('adver-v0', target_label=6, test_mode=False)

    callbacks = [ModelIntervalCheckpoint(DQN_WEIGHTS_FILENAME, interval=WEIGHTS_CHECKPOINT)]
    callbacks += [FileLogger(DQN_LOG_FILENAME, interval=LOG_CHECKPOINT)]
    dqn = init_dqn(fit_steps)
    if cont:
        dqn.load_weights(DQN_WEIGHTS_FILENAME.format(step="final"))
        print("continue fitting from last checkpoint")

    dqn.fit(training_env, callbacks=callbacks, nb_steps=fit_steps, start_step_policy=start_policy, nb_max_start_steps=3)

    dqn.save_weights(DQN_WEIGHTS_FILENAME.format(step="final"), overwrite=True)


@click.command()
@click.argument('nb_episodes', type=int)
def test_dqn(nb_episodes: int):
    testing_env = gym.make('adver-v0', target_label=6, test_mode=True)
    dqn = init_dqn(1000)
    dqn.load_weights(DQN_WEIGHTS_FILENAME.format(step="final"))
    dqn.test(testing_env, nb_episodes=nb_episodes, visualize=True)


@click.group()
def cli():
    pass


cli.add_command(fit_dqn)
cli.add_command(test_dqn)


if __name__ == '__main__':
    cli()


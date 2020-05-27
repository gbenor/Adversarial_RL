from collections import namedtuple

import gym
# import pandas as pd
import numpy as np
from gym import spaces
from keras.utils import to_categorical
from numpy import linalg as LA

from gym_adversarial.envs.centers import Centers
from gym_adversarial.envs.consts import ACTIONS, INITIAL_STEP_SIZE, MAX_PERTURBATION, MAX_STEPS, MIN_STEPS_REWARD_TH0, \
    MIN_STEPS_REWARD_TH1, NUM_CLASSES, SAMPLES_FOR_CALC_CENTERS
from gym_adversarial.envs.conv_mnist import ConvMinst
from gym_adversarial.utils import load_data


def perturbation_reward(cur_perturbation, max_perturbation):
    return max(0, (max_perturbation - cur_perturbation))


def label_reward(predicted_values, target_label):
    target_score = predicted_values[target_label]
    label_score = 1 if np.argmax(predicted_values) == target_label else 0
    return label_score
    # return target_score + label_score


def min_steps_reward(steps):
    if steps < MIN_STEPS_REWARD_TH0:
        return 1
    if steps < MIN_STEPS_REWARD_TH1:
        return 0
    return (MIN_STEPS_REWARD_TH1 - steps) / steps


def select_initial_sample(samples, labels, target_label):
    while True:
        index = np.random.choice(len(samples))
        if labels[index] != target_label:
            break
    return samples[index].reshape(28, 28, 1), labels[index]


class AdversarialMNIST(gym.Env):
    """An environment for finding adversarial examples in MNIST for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, target_label=6, step_size: float = INITIAL_STEP_SIZE, test_mode = False):
        super(AdversarialMNIST, self).__init__()
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(low=0, high=1, shape=(28, 28), dtype=np.float32)
        self.step_size = step_size
        self.test_mode = test_mode

        self.train_images, self.train_labels, self.test_images, self.test_labels = load_data()
        model_file = "gym-adversarial/gym_adversarial/envs/conv_mnist_model"
        cluster_file = "gym-adversarial/gym_adversarial/envs/cluster.pkl"
        center_file = "gym-adversarial/gym_adversarial/envs/center.pkl"

        self.classifier = ConvMinst(model_file,
                                    self.train_images, to_categorical(self.train_labels, NUM_CLASSES),
                                    self.test_images, to_categorical(self.test_labels, NUM_CLASSES))
        # self.cluster = MNIST_Cluster(cluster_file, NUM_OF_CLUSTERS, self.train_images, self.train_labels)
        self.cluster = Centers(fname=center_file, k=SAMPLES_FOR_CALC_CENTERS, target_label=target_label,
                               samples=self.train_images, labels=self.train_labels)
        self.target_label = target_label
        # fig = plt.figure()
        # fig.add_axes([0, 0, 0.5, 1.0])
        # self.ax1 = fig.axes[0]
        # self.ax1.set_yticklabels([])
        # self.ax1.set_xticklabels([])

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        if self.test_mode:
            sample, label = select_initial_sample(self.test_images, self.test_labels, self.target_label)
        else:
            sample, label = select_initial_sample(self.train_images, self.train_labels, self.target_label)

        self.orig_image = np.copy(sample)
        self.orig_label = np.copy(label)
        self.cur_image = np.copy(sample)
        return self._next_observation()

    def render(self, mode='human', close=False):
        obs = self._next_observation()
        print(f"Original label={obs.original_label}\t Current label={np.argmax(obs.predicted_labels)}\t"
              f"Target prediction score={obs.predicted_labels[self.target_label]:.2f}\t Perturbation={obs.perturbation_norm:.2f}\t"
              f"Step size={self.step_size:.4f}\t")
        # self.ax1.clear()
        # self.ax1.imshow(self.cur_image, cmap='gray_r')
        # plt.draw()
        # plt.pause(0.001)

    def _next_observation(self):
        Observation = namedtuple('Observation', ['image', 'original_label', 'predicted_labels',
                                                 'perturbation', 'perturbation_norm'])
        predicted_labels = self.classifier._model(self.cur_image.reshape(1, 28, 28, 1)).numpy().ravel()
        perturbation = self.cur_image.reshape(28, 28) - self.orig_image.reshape(28, 28)
        perturbation_norm = LA.norm(perturbation, ord="fro")
        return Observation(self.cur_image, self.orig_label, predicted_labels, perturbation, perturbation_norm)

    def _take_action(self, action: str):
        assert action in ACTIONS, f"illegal action: {action}"
        # print(f"Action = {action}")
        direction = None
        if action == "CLOSET_CLUSTER":
            direction = self.cluster.get_closest_center(self.cur_image)
        if action == "FARTHEST_CLUSTER":
            direction = self.cluster.get_farthest_center(self.cur_image)
        if action == "ORIGINAL_IMAGE":
            direction = self.orig_image
        if action == "DECREASE_STEP":
            self.step_size /= 2
            direction = 0

        self.cur_image += self.step_size * direction

    def step(self, action: int):
        # Execute one time step within the environment
        self.current_step += 1
        self._take_action(ACTIONS[action])
        obs = self._next_observation()

        if label_reward(obs.predicted_labels, self.target_label)>0:
            reward = 1 + 100*perturbation_reward(obs.perturbation_norm, MAX_PERTURBATION)
        else:
            reward = 0

        #
        # reward = REWARD_COEF["PERTURBATION"] * perturbation_reward(obs.perturbation_norm, MAX_PERTURBATION) + \
        #          REWARD_COEF["LABEL"] * label_reward(obs.predicted_labels, self.target_label) + \
        #          REWARD_COEF["MIN_STEPS"] * min_steps_reward(self.current_step)
        #
        # # delay_modifier = max((self.current_step / MAX_STEPS), 1)
        # delay_modifier = 1
        # reward = reward * delay_modifier
        done = (self.current_step == MAX_STEPS) or \
               (label_reward(obs.predicted_labels, self.target_label) > 0 and
                perturbation_reward(obs.perturbation_norm, MAX_PERTURBATION) > 0)

        return obs, reward, done, {}

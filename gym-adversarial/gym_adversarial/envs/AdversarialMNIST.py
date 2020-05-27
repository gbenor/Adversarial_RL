from collections import namedtuple

import gym
import numpy as np
from gym import spaces
from keras.utils import to_categorical
from numpy import linalg as LA

from gym_adversarial.envs.centers import Centers
from gym_adversarial.envs.consts import ACTIONS, INITIAL_STEP_SIZE, MAX_PERTURBATION, MAX_STEPS, NUM_CLASSES, \
    SAMPLES_FOR_CALC_CENTERS, MIN_PERTURBATION_REWARD, MAX_PERTURBATION_REWARD, \
    CLASSIFIER_MODEL_FILE, CENTER_FILE, REWARD_COEF, STEPS_TO_IMPROVE
from gym_adversarial.envs.conv_mnist import ConvMinst
from gym_adversarial.utils import load_data


def perturbation_reward(cur_perturbation, max_perturbation):
    return np.clip((max_perturbation - cur_perturbation),
                   a_min=MIN_PERTURBATION_REWARD, a_max=MAX_PERTURBATION_REWARD)


def reach_target_label(predicted_values, target_label) -> bool:
    return np.argmax(predicted_values) == target_label


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

    def __init__(self, target_label=6, step_size: float = INITIAL_STEP_SIZE, test_mode=False):
        super(AdversarialMNIST, self).__init__()
        self.target_label = target_label
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(low=0, high=1, shape=(28, 28), dtype=np.float32)
        self.step_size = step_size
        self.test_mode = test_mode

        self.train_images, self.train_labels, self.test_images, self.test_labels = load_data()

        self.classifier = ConvMinst(CLASSIFIER_MODEL_FILE,
                                    self.train_images, to_categorical(self.train_labels, NUM_CLASSES),
                                    self.test_images, to_categorical(self.test_labels, NUM_CLASSES))
        # self.cluster = MNIST_Cluster(CLUSTER_FILE, NUM_OF_CLUSTERS, self.train_images, self.train_labels)
        self.cluster = Centers(fname=CENTER_FILE, k=SAMPLES_FOR_CALC_CENTERS, target_label=target_label,
                               samples=self.train_images, labels=self.train_labels)

        self.orig_image = None
        self.orig_label = None
        self.cur_image = None
        self.current_step = None
        self.current_step_while_acceptable_result = None
        self.last_action = None

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        self.current_step_while_acceptable_result = 0
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
        print(
            f"Step={self.current_step}\t Original label={obs.original_label}\t "
            f"Curr label={np.argmax(obs.predicted_labels)}\t"
            f"Target pred score={obs.predicted_labels[self.target_label]:.2f}\t "
            f"Pert={obs.perturbation_norm:.2f}\t "
            f"Step size={self.step_size:.4f}\t"
            f"Last act={self.last_action}")

    def _next_observation(self):
        Observation = namedtuple('Observation', ['image', 'original_label', 'predicted_labels',
                                                 'perturbation', 'perturbation_norm'])
        predicted_labels = self.classifier._model(self.cur_image.reshape(1, 28, 28, 1)).numpy().ravel()
        perturbation = self.cur_image.reshape(28, 28) - self.orig_image.reshape(28, 28)
        perturbation_norm = LA.norm(perturbation, ord="fro")
        return Observation(self.cur_image, self.orig_label, predicted_labels, perturbation, perturbation_norm)

    def _take_action(self, action: str):
        assert action in ACTIONS, f"illegal action: {action}"
        self.last_action = action
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

        # reward only if we reach the target label
        reward = 0
        if reach_target_label(obs.predicted_labels, self.target_label):
            reward = 1 * REWARD_COEF["LABEL"] + \
                     REWARD_COEF["PERTURBATION"] * perturbation_reward(obs.perturbation_norm, MAX_PERTURBATION)

        acceptable_result: bool = reach_target_label(obs.predicted_labels, self.target_label) and \
                                  (perturbation_reward(obs.perturbation_norm, MAX_PERTURBATION) >= 0)

        if acceptable_result:
            self.current_step_while_acceptable_result += 1

        done = (self.current_step == MAX_STEPS) or (self.current_step_while_acceptable_result == STEPS_TO_IMPROVE)

        return obs, reward, done, {}

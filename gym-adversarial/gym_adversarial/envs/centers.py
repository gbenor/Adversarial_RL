from itertools import combinations

import numpy as np
from pathlib import Path
import pickle
from gym_adversarial.utils import extract_samples_by_label, select_random_samples
from numpy import linalg as LA


def calc_distance(s1, s2):
    diff = s1.reshape(28, 28) - s2.reshape(28, 28)
    return LA.norm(diff, ord="fro")


class Centers():
    def __init__(self, fname, k, target_label, samples, labels):
        self._fname = Path(fname)
        self.model = None
        if self._fname.exists():
            self.load()
        else:
            self.fit(samples, labels, k, target_label)
            self.save()

    def save(self):
        with self._fname.open("wb")  as handle:
            pickle.dump(self.model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self):
        with self._fname.open("rb")  as handle:
            self.model = pickle.load(handle)
        print("cluster: load pretrained cluster")

    def fit(self, samples, labels, k, target_label):
        samples = extract_samples_by_label(samples, labels, target_label)
        k_samples = select_random_samples(samples, k)
        v1, v2 = None, None
        max_dist = 0
        for i, j in combinations(range(len(k_samples)), 2):
            d = calc_distance(k_samples[i], k_samples[j])
            if d > max_dist:
                v1, v2 = k_samples[i], k_samples[j]
                max_dist = d

        assert max_dist > 0
        self.model = (v1.reshape(28,28,1), v2.reshape(28,28,1))

    def get_closest_center(self, x):
        if calc_distance(self.model[0], x) < calc_distance(self.model[1], x):
            return self.model[0]
        return self.model[1]

    def get_farthest_center(self, x):
        if calc_distance(self.model[0], x) > calc_distance(self.model[1], x):
            return self.model[0]
        return self.model[1]




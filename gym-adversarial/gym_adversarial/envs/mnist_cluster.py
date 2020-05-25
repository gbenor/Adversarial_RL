import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path
import pickle

class MNIST_Cluster():
    def __init__(self, fname, k, sampels, labels):
        self._fname = Path(fname)
        if self._fname.exists():
            self.load()
        else:
            self.fit(sampels, labels, k)

    def save(self):
        with self._fname.open("wb")  as handle:
            pickle.dump(self.kmeans, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self):
        with self._fname.open("rb")  as handle:
            self.kmeans = pickle.load(handle)
        print("cluster: load pretrained cluster")

    def fit(self, samples, labels, k):
        unique_labels = np.unique(labels)
        self.kmeans = {}
        for label in unique_labels:
            print(f"cluster: fit label {label}")
            idx = np.argwhere(labels == label)
            label_samples = samples[idx]
            label_samples = label_samples.reshape(len(label_samples), -1)
            self.kmeans[label] = KMeans(k)
            self.kmeans[label].fit(label_samples)
        self.save()

    def get_closest_center(self, x, label):
        d = self.kmeans[label].transform(x.reshape(1,-1))
        return d, self.kmeans[label].cluster_centers_[np.argmin(d)].reshape(28,28,1)

    def get_farthest_center(self, x, label):
        d = self.kmeans[label].transform(x.reshape(1,-1))
        return d, self.kmeans[label].cluster_centers_[np.argmax(d)].reshape(28,28,1)





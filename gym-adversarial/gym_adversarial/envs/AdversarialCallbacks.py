import pickle
from pathlib import Path
from rl.callbacks import Callback
import numpy as np


def generate_filename(target, minpert, minpert_step):
    return f"target={target}_minpert={minpert}_minpert_step={minpert_step}"


def find_minpert_index(history, target):
    images_labeled_as_target = np.array(history["labels"]) == target
    min_pert = (np.array(history["perturbations"])[images_labeled_as_target]).min()
    index = np.where(np.array(history["perturbations"])==min_pert)[0][0]
    assert history["labels"][index] == target, "maybe two images has the same perturbation value"
    return index


class CustomHistoryCallback(Callback):
    def __init__(self):
        super().__init__()
        self.history = None

    def on_episode_begin(self, episode, logs={}):
        """Called at beginning of each episode"""
        self.history = {
            "target": self.env.target_label,
            "source_label": self.env.orig_label,
            "images": [],
            "labels": [],
            "perturbations": [],
            "actions" : []
        }

    def on_action_begin(self, action, logs={}):
        self.history["actions"].append(action)


    def on_step_begin(self, step, logs={}):
        """Called at end of each step"""
        obs = self.env._next_observation()

        self.history["images"].append(np.copy(obs.image))
        self.history["labels"].append(np.argmax(obs.predicted_labels))
        self.history["perturbations"].append(obs.perturbation_norm)

    def on_episode_end(self, episode, logs={}):
        """Called at end of each episode"""
        result_directory = Path(self.env.result_directory)
        result_directory.mkdir(parents=True, exist_ok=True)

        target = self.history["target"]
        result_file = result_directory / f"{self.env.test_description}_target={target}_episode={episode}.pkl"
        with result_file.open("wb") as handle:
            pickle.dump(self.history, handle, protocol=pickle.HIGHEST_PROTOCOL)





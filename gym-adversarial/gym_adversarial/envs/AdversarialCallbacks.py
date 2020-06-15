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
            "images": [],
            "labels": [],
            "perturbations": []
        }

    def on_step_end(self, step, logs={}):
        """Called at end of each step"""
        print("on_step_end callback")
        obs = self.env._next_observation()

        self.history["images"].append(obs.image)
        self.history["labels"].append(np.argmax(obs.predicted_labels))
        self.history["perturbations"].append(obs.perturbation_norm)

    def on_episode_end(self, episode, logs={}):
        """Called at end of each episode"""
        print("on_episode_end")
        print(" ***** min pert: ******")
        i = find_minpert_index(self.history, self.env.target_label)
        print(i)
        print (self.history["labels"][i])
        print (self.history["perturbations"][i])





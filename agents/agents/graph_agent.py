import numpy as np
from rl.core import Agent

# # class Agent(object):
#     """Abstract base class for all implemented agents.
#     Each agent interacts with the environment (as defined by the `Env` class) by first observing the
#     state of the environment. Based on this observation the agent changes the environment by performing
#     an action.
#     Do not use this abstract base class directly but instead use one of the concrete agents implemented.
#     Each agent realizes a reinforcement learning algorithm. Since all agents conform to the same
#     interface, you can use them interchangeably.
#     To implement your own agent, you have to implement the following methods:
#     - `forward`
#     - `backward`
#     - `compile`
#     - `load_weights`
#     - `save_weights`
#     - `layers`
#
from gym_adversarial.envs.consts import STEPS_TO_CHANGE_CLUSTER


class GraphAgent(Agent):
    def __init__(self, target_class, policy_graph, init_state=None):
        super().__init__()
        self.target_class = target_class
        self.policy_graph = policy_graph
        self.init_state = init_state
        self.current_state = None
        self.steps_to_change_cluster = None

        self.compiled = True

    def reset_states(self):
        """Resets all internally kept states after an episode is completed.
        """
        self.current_state = self.init_state
        self.steps_to_change_cluster = STEPS_TO_CHANGE_CLUSTER
        print ("episode reset_states")

    def forward(self, observation):
        """Takes the an observation from the environment and returns the action to be taken next.
        If the policy is implemented by a neural network, this corresponds to a forward (inference) pass.
        # Argument
            observation (object): The current observation from the environment.
        # Returns
            The next action to be executed in the environment.
        """
        assert self.current_state is not None, "No init state"
        print (f"forward: {self.current_state}")
        node_data = self.policy_graph.nodes[self.current_state]
        successor = next(self.policy_graph.successors(self.current_state))
        predicted_class = np.argmax(observation.predicted_labels)
        predict_target_class = predicted_class == self.target_class

        if self.current_state == "ChangeCenters":
            if observation.steps > self.steps_to_change_cluster:
                self.steps_to_change_cluster+=STEPS_TO_CHANGE_CLUSTER
                self.current_state = self.init_state
                return node_data['action']
            else:
                self.current_state = successor
                return self.forward(observation)

        if node_data['action'] is None:
            self.current_state = successor
            return self.forward(observation)

        if node_data['action'] == "DECREASE_STEP":
            self.current_state = successor
            return "DECREASE_STEP"

        assert node_data['stop_on_target'] is not None
        if node_data['stop_on_target']:
            if predict_target_class:
                self.current_state = successor
                return self.forward(observation)
        else:
            # not stop on target
            if not predict_target_class:
                self.current_state = successor
                return self.forward(observation)

        return node_data['action']

    @property
    def layers(self):
        raise NotImplementedError

    def save_weights(self, filepath, overwrite=False):
        raise NotImplementedError

    def load_weights(self, filepath):
        raise NotImplementedError

    def compile(self, optimizer, metrics=[]):
        pass

    def backward(self, reward, terminal):
        pass

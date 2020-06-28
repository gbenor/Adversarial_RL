#########################################
# File path
#########################################
CLASSIFIER_MODEL_FILE = "gym-adversarial/gym_adversarial/envs/conv_mnist_model"
CLUSTER_FILE = "gym-adversarial/gym_adversarial/envs/cluster.pkl"
CENTER_FILE = "gym-adversarial/gym_adversarial/envs/center.pkl"
DQN_DIR = "gym-adversarial/gym_adversarial/envs/"
DQN_WEIGHTS_FILENAME = "gym-adversarial/gym_adversarial/envs/dqn_weights_small_action_set_{step}.h5f"
WEIGHTS_CHECKPOINT = 10000
DQN_LOG_FILENAME = "gym-adversarial/gym_adversarial/envs/dqn_log.json"
LOG_CHECKPOINT = 1000

#########################################
# Environment parameters
#########################################
# ACTIONS = ["CLOSET_CLUSTER", "FARTHEST_CLUSTER", "ORIGINAL_IMAGE", "DECREASE_STEP"]
ACTIONS = ["CLOSET_CLUSTER", "FARTHEST_CLUSTER", "ORIGINAL_IMAGE", "DECREASE_STEP", "INCREASE_STEP", "NEW_CENTERS"]
NB_ACTIONS = len(ACTIONS)

REWARD_COEF = {
    "PERTURBATION": 10,
    "LABEL": 1,
}
MAX_PERTURBATION = 5.0
MIN_PERTURBATION_REWARD = -2
MAX_PERTURBATION_REWARD = 2

INITIAL_STEP_SIZE = 2
MAX_STEPS = 2000
STEPS_TO_IMPROVE = 3
MIN_STEP_SIZE = 0.05
MAX_STEP_SIZE = 2

MIN_DIFF_BETWEEN_IMAGES = 0.001

NUM_CLASSES = 10
NUM_OF_CLUSTERS = 2
SAMPLES_FOR_CALC_CENTERS = 300

STEPS_TO_CHANGE_CLUSTER = 200

#########################################
# DQN parameters
#########################################
# Memory
DQN_LR = 0.01 #.00025
WINDOW_LENGTH = 4
MEMORY_LIMIT = 10000

TARGET_MODEL_UPDATE = 500

# MIN_STEPS_REWARD_TH0 = 5
# MIN_STEPS_REWARD_TH1 = 10


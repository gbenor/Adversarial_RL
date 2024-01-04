# Adversarial Gym: A Reinforcement-Learning Framework for Designing and Evaluating Adversarial Attacks

## Abstract

Adversarial examples are maliciously manipulated inputs that deceive machine learning models. This project presents "Adversarial Gym," a framework based on the GYM environment for planning and testing Decision-Based (D.B.) attacks. The framework, in conjunction with the CleverHans environment, aims to enhance research cycles and inspire new algorithms, particularly in reinforcement learning. This work introduces a novel deterministic algorithm and a reinforcement learning algorithm to deceive an MNIST classifier, both of which are tested using this framework.


## Introduction

This project explores the susceptibility of high-performance machine learning algorithms to adversarial perturbations. It focuses on decision-based attacks, a form of black box attacks that depend solely on the model's final decision. The project develops a decision-based attack environment based on the OpenAI Reinforcement Learning Gym interface, suitable for the MNIST dataset and expandable for other datasets.

## Key Contributions

- A research framework for decision-based attacks.
- Custom actions for attack algorithms.
- Design, implementation, and evaluation of two boundary attack algorithms.
- Training and evaluation of a reinforcement learning agent in this environment.

## Installation and Usage

1. Clone the repository: `git clone https://github.com/gbenor/Adversarial_RL`.
2. Follow the installation instructions provided in the documentation to set up the environment.
3. Use the provided scripts and examples to test and develop new adversarial attack strategies.

## Research Background

Adversarial Gym leverages reinforcement learning to create adversarial examples with the aim of understanding and improving the integrity and security of machine learning algorithms. This project is rooted in both the necessity for robust defense mechanisms in ML and the academic pursuit of understanding the difference between human and computer sensory information processing.

## Future Directions

This framework sets the stage for further exploration in the field of adversarial machine learning, including expanding action spaces, refining reinforcement learning algorithms, and integrating this framework with existing tools like CleverHans for comprehensive adversarial attack development.

## Acknowledgments

Special thanks to the authors and contributors of referenced works in the field of adversarial machine learning and reinforcement learning. This project builds upon their foundational research and aims to contribute further to the community.

## References

- Detailed references are provided within the project documentation, acknowledging all the foundational works and papers that have inspired this project.

## License

Adversarial Gym is freely available for use and modification.

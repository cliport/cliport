# Model Card: CLIPort

Following [OpenAI's CLIP (Radford et al.)](https://github.com/openai/CLIP/blob/main/model-card.md), [Model Cards for Model Reporting (Mitchell et al.)](https://arxiv.org/abs/1810.03993) and [Lessons from Archives (Jo & Gebru)](https://arxiv.org/pdf/1912.10389.pdf) we provide additional information on CLIPort.

## Model Details


### Overview
- Developed by Shridhar et al. at University of Washington and NVIDIA. CLIPort is an end-to-end imitation-learning agent that can learn a single language-conditioned policy for various tabletop tasks. The framework combines the broad semantic understanding (_what_) of [CLIP](https://openai.com/blog/clip/) with the spatial precision (_where_) of [TransporterNets](https://transporternets.github.io/) to learn generalizable skills from limited training demonstrations. See: [cliport.github.io](https://cliport.github.io).
- Fully Convolutional Networks trained with end-to-end supervised learning.
- Trained for pick-and-place tabletop manipulation tasks where objects appear on a planar surface.

### Model Date

October 2021

### Documents

- [CLIPort Paper](https://arxiv.org/abs/2109.12098)
- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [TransporterNets Paper](https://arxiv.org/abs/2010.14406)


## Model Use

- **Primary intended use case**: CLIPort is intended for robotic manipulation research. We hope the benchmark and pre-trained models will enable researchers to study the generalization capabilities of end-to-end manipulation frameworks. Specifically, we hope the setup serves a reproducible framework for evaluating  robustness and scaling capabilities of manipulation agents. 
- **Primary intended users**: Robotics researchers. 
- **Out-of-scope use cases**: Deployed use cases in real-world autonomous systems without human supervision is currently out-of-scope. Use cases that involve manipulating novel objects without humans-in-the-loop is also not recommended for safety-critical systems. The agent is also intended to be trained and evaluated with English language instructions.

## Data

- Pre-training Data for CLIP: See [OpenAI's Model Card](https://github.com/openai/CLIP/blob/main/model-card.md#data) for full details.
- Manipulation Data for CLIPort: The agent was trained with image-caption-action pairs from expert demonstrations. In simulation we use oracle agents and in real-world we use human demonstrations. Since the agent is used in few-shot settings with very limited data, the agent might exploit intended and un-intented biases in the training demonstrations. Currently, these biases are limited to just objects that appear on tabletops.


## Limitations

- Limited to SE(2) action space.
- Exploits biases in training demonstrations.
- Needs good hand-eye calibration.
- Struggles with novel objects that are completely outside the training distribution of objects.
- Struggles with grounding complex spatial relationships. 
- Does not predict task completion.
- Prone to biases in CLIP's training data.

See Appendix I in the [paper](https://arxiv.org/abs/2109.12098) for an extended discussion.
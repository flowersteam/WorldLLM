# WorldLLM
We introduce WorldLLM, a framework that aims to transform a language model into a world model, by improving its understanding of the environment and its ability to predict its evolution. The method developed consists of searching for and constructing a set of explanations, which are then transmitted to the language model. We take inspiration from prior work by collecting trajectories with an agent rewarded to collect transitions that the model struggles to predict.

This work is still opened research and aims to better understand and use LLMs in reinforcement learning environments.


## Method
![WorldLLM](Overview.png)   
The method is composed of three modules:
1. **The Experimenter** : An agent(RL or LLM) whose role is to interact with the environment and gather experiences.
2. **The Theorist**: An LLM with good reasoning skills, who must extract information from the different trajectories to hypothesize rules about the environment.
3. **The Statistician** : The LLM to be transformed into a world model. Its objective is to predict the evolution of the environment based on the information contained in the explanations provided. It is also used to assess the quality of the explanations and to determine whether or not more explanations are needed.

More details will be available when the paper will be accessible.

## Project Structure

The repository is organized as follows:

```
WorldLLM/
│
├── configs/
│   ├── algorithm/
│   ├── environment/
│   ├── experimenter/
│   ├── llm/
│   ├── base_config.yaml
│   └── ...
│
├── lab/
│
├── montecarlo_methods/
│   ├── importance_sampling.py
│   └── metropolis_hastings.py
│
├── utils/
│
├── worldllm_envs/
│   ├── door/
│   └── playground/
│
├── main.py
├── .gitignore
├── README.md
└── LICENSE
```

- `configs/`: YAML file to configure main.py using `hydra`
- `lab/`: Folder containing single files to evaluate the generated worldmodel or to test scoring methods
- `montecarlo_methods/`: Implementation of Monte Carlo methods.
- `utils/`: Utility functions and scripts for LLMs and agents
- `worldllm_envs/`: Custom environments for the WorldLLM project.
- `main.py`: Main script to run the project.

# Getting Started
This project was developed on PyTorch using the Hugging Face's Transformers library.
## Setup

The project was run with Python 3.10.12 and managed in a virtual environment with pip. To run this repository, ensure you have all the required dependencies installed by running:

```sh
pip install -r requirements.txt
```
You also need to install worldllm_envs as a module:
```sh
pip install -e worldllm_envs
```
## Run

The configuration of the script is managed with `hydra`. To obtain the intended behavior, create a YAML file based on the examples in `./configs/`. To launch the program, execute the following command:

```sh
python main.py -cn config_name
```
For example
```sh
python main.py -cn play_metropolis_pb.yaml
```
## Configs
Each config looks similar to this:
```yaml
defaults:
  - base_config                                 # Contain base config(like seed, exp_name)
  - algorithm: metropolis_hastings              # Monte Carlo Method to use
  - llm@theorist: Phi-3-mini-4k-instruct        # Theorist LLM
  - llm@statistician: Phi-3-mini-4k-instruct    # Statistician LLM
  - experimenter: sb3agent                      # Experimenter model to use(oracle or RL agent)
  - environment: playground_rl                  # Environment to use

exp_name: test_pipeline                         # Override defaut exp name

statistician: null                              # Override to null to use the same LLM as the Theorist
seed: 60                                        # Seed to use
algorithm:                                      # Override the algorithm base config 
  nb_phases: 400
  nb_iterations: 5
```

# FAQ
If you have any questions, open an issue. I will try to answer them the best I can.
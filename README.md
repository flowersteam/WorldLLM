# WorldLLM
We introduce WorldLLM, a framework that aims to transform a language model into a world model, by improving its understanding of the environment and its ability to predict its evolution. The method developed consists of searching for and constructing a set of explanations, which are then transmitted to the language model. We take inspiration from prior work by collecting trajectories with an agent rewarded to collect transitions that the model struggles to predict.

This work is still opened research and aims to better understand and use LLMs in reinforcement learning environments.


## Method
The method is composed of three modules:
1. **The Experimenter** : An agent(RL or LLM) whose role is to interact with the environment and gather experiences.
2. **The Theorist**: An LLM with good reasoning skills, who must extract information from the different trajectories to hypothesize rules about the environment.
3. **The Statistician** : The LLM to be transformed into a world model. Its objective is to predict the evolution of the environment based on the information contained in the explanations provided. It is also used to assess the quality of the explanations and to determine whether or not more explanations are needed.
![WorldLLM](Overview.png)

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

## Run

The configuration of the script is managed with `hydra`. To obtain the intended behavior, create a YAML file based on the examples in `./configs/`. To launch the program, execute the following command:

```sh
python main.py -cn <config_name>
```
For example
```sh
python main.py -cn play_metropolis_pb.yaml
```

## FAQ
If you have any questions, open an issue. I will try to answer them the best I can.
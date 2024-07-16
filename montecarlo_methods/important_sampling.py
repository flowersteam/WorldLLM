from inspect import indentsize
from typing import Tuple

import numpy as np
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils_env import BaseAgent, generate_text_trajectories
from utils_llm import compute_likelihood, generate_rules
from worldllm_envs.envs.base import BaseRuleEnv


def important_sampling(
    env: BaseRuleEnv,
    agent: BaseAgent,
    theorist: Tuple[AutoModelForCausalLM, AutoTokenizer],
    statistician: Tuple[AutoModelForCausalLM, AutoTokenizer],
    cfg: DictConfig,
) -> None:
    # Define true rule
    true_rule = env.generate_rule()
    # Init weights
    weights = np.ones((cfg.nb_rules, 1)) / cfg.nb_rules

    # Generate trajectories
    prompt_trajectories = generate_text_trajectories(
        env, agent, true_rule, cfg.nb_trajectories
    )
    # Sample rules
    rules, importance_probs = generate_rules(
        theorist, prompt_trajectories, cfg.nb_rules
    )
    if cfg.add_true_rule:
        rules.append(true_rule)
        importance_probs = np.append(importance_probs, np.log(1 / (cfg.nb_rules + 1)))

    # Compute likelihoods of new data using the rules
    likelihoods = compute_likelihood(statistician, rules, prompt_trajectories)

    # weights is just the likelihoods for importance sampling with resampling
    weights = likelihoods

    # Print rules and weights sorted
    indices = np.argsort(-weights)
    print("------------------------")
    print("true rule: " + repr(true_rule))
    print("------------------------")
    for ind in indices:
        print("-----rule-----:   " + repr(rules[ind]) + f" weight: {weights[ind]:2f}")

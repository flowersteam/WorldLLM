from typing import List, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils_env import BaseAgent, generate_text_trajectories
from utils_llm import compute_likelihood, generate_rules
from worldllm_envs.envs.base import BaseRuleEnv


def update_weights(
    weights: np.ndarray, log_posterior: np.ndarray, log_likelihood: np.ndarray
) -> np.ndarray:
    """Update the weights using the classic importance sampling formula and normalize them."""
    weights = log_likelihood - log_posterior
    return weights


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
    # Compute likelihoods of new data using the rules
    likelihoods = compute_likelihood(statistician, rules, prompt_trajectories)

    # Update weights
    weights = update_weights(weights, importance_probs, likelihoods)

    # Print rules and weights
    print("------------------------")
    print("true rule: " + repr(true_rule))
    print("------------------------")
    for rule, weight in zip(rules, weights):
        print("-----rule-----:   " + repr(rule) + f": {weight}")

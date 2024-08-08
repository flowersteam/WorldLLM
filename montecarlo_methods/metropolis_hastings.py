from copy import copy
from typing import Dict, List, Tuple

import numpy as np
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils_env import BaseAgent, generate_text_trajectories
from utils_llm import compute_likelihood, evolve_rules, generate_rules
from worldllm_envs.envs.base import BaseRuleEnv


def metropolis_hastings(
    env: BaseRuleEnv,
    agent: BaseAgent,
    theorist: Tuple[AutoModelForCausalLM, AutoTokenizer],
    statistician: Tuple[AutoModelForCausalLM, AutoTokenizer],
    cfg: DictConfig,
) -> None:
    """Metropolis-Hasting algorithm"""
    # Define true rule
    true_rule = env.generate_rule()

    # Generate trajectories
    prompt_trajectories = generate_text_trajectories(
        env, agent, true_rule, cfg.nb_trajectories
    )
    # Sample rules
    prev_rules, prev_importance_probs = generate_rules(
        theorist, prompt_trajectories, cfg.nb_rules
    )
    prev_likelihoods = compute_likelihood(statistician, prev_rules, prompt_trajectories)
    all_rules = copy(prev_rules)
    all_importance_probs = prev_importance_probs
    all_likelihoods = prev_likelihoods
    all_prev_rules_ind = [-1] * cfg.nb_rules
    prev_rules_ind = np.zeros((cfg.nb_rules,), dtype=int)
    for i in range(cfg.nb_iterations):
        # Sample a new rule
        rules, importance_probs = evolve_rules(
            theorist, prompt_trajectories, prev_rules
        )
        # Compute likelihoods of new data using the rules
        likelihoods = compute_likelihood(statistician, rules, prompt_trajectories)
        weights = (
            likelihoods - prev_likelihoods + importance_probs - prev_importance_probs
        )
        # Update rules obtained
        all_rules.extend(rules)
        all_importance_probs = np.append(all_importance_probs, importance_probs)
        all_likelihoods = np.append(all_likelihoods, likelihoods)
        all_prev_rules_ind.extend(prev_rules_ind)
        # Accept or reject
        mask = np.where(np.log(np.random.rand()) < weights, 1, 0)
        prev_rules_ind = np.where(mask, i, prev_rules_ind)
        prev_rules = np.where(mask, rules, prev_rules)
        prev_likelihoods = np.where(mask, likelihoods, prev_likelihoods)
        prev_importance_probs = np.where(mask, importance_probs, prev_importance_probs)
    indices = np.argsort(-np.array(all_likelihoods))
    print("------------------------")
    print("true rule: " + repr(true_rule))
    print("------------------------")
    for ind in indices:
        print(
            f"-----rule-----:{ind}({all_prev_rules_ind[ind]}):   {repr(all_rules[ind])}, importance: {all_importance_probs[ind]:2f}, likelihood: {all_likelihoods[ind]:2f}"
        )

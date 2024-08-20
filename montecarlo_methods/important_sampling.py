from typing import Dict, List, Tuple

import numpy as np
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.utils_env import BaseAgent, generate_text_trajectories
from utils.utils_llm import compute_likelihood, generate_rules
from utils.utils_save import RuleOutput
from worldllm_envs.envs.base import BaseRuleEnv


def get_unique_rules(
    rules: List[str], weights: np.ndarray
) -> Tuple[Dict[str, int], np.ndarray, np.ndarray]:
    """From rules and weights return unique rules with count and the weights with correct index"""
    set_rules = {}
    unique_rules = []
    counts = []
    new_weights = []
    for i, rule in enumerate(rules):
        if rule not in unique_rules:
            set_rules[rule] = len(unique_rules)
            unique_rules.append(rule)
            counts.append(1)
            new_weights.append(weights[i])
        else:
            counts[set_rules[rule]] += 1
    return unique_rules, np.array(counts), np.array(new_weights)


def important_sampling(
    env: BaseRuleEnv,
    agent: BaseAgent,
    theorist: Tuple[AutoModelForCausalLM, AutoTokenizer],
    statistician: Tuple[AutoModelForCausalLM, AutoTokenizer],
    cfg: DictConfig,
) -> RuleOutput:
    # Get true rule
    true_rule = env.rule

    # Generate trajectories
    prompt_trajectories = generate_text_trajectories(
        env, agent, true_rule, cfg.nb_trajectories
    )
    # Sample rules
    rules, importance_probs = generate_rules(
        theorist, prompt_trajectories, cfg.nb_rules
    )
    # Get unique rules and counts
    rules, counts, importance_probs = get_unique_rules(rules, importance_probs)
    if cfg.add_true_rule:
        rules.append(true_rule)
        counts = np.append(counts, 1)
        importance_probs = np.append(importance_probs, np.log(1 / (cfg.nb_rules + 1)))

    # Compute likelihoods of new data using the rules
    likelihoods = compute_likelihood(statistician, rules, prompt_trajectories)

    # weights is just the likelihoods for importance sampling with resampling
    weights = np.log(counts) + likelihoods - importance_probs

    # Print rules and weights sorted
    indices = np.argsort(-likelihoods)
    print("------------------------")
    print("true rule: " + repr(true_rule))
    print("------------------------")
    for ind in indices:
        print(
            f"-----rule-----:   {repr(rules[ind])} weight: {weights[ind]:2f}, importance: {importance_probs[ind]:2f}, likelihood: {likelihoods[ind]:2f}, count: {counts[ind]}"
        )
    return RuleOutput(
        true_rule,
        rules,
        likelihoods,
        {
            "weights": weights,
            "counts": counts,
            "importance_probs": importance_probs,
            "likelihoods": likelihoods,
        },
    )

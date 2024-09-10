from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from utils.utils_env import BaseAgent, generate_text_trajectories
from utils.utils_llm import LlmModel, compute_likelihood, generate_rules
from utils.utils_save import RuleOutput
from worldllm_envs.base import BaseRuleEnv


def get_unique_rules(
    rules: List[str], weights: np.ndarray
) -> Tuple[List[str], np.ndarray, np.ndarray]:
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
    theorist: LlmModel,
    statistician: LlmModel,
    cfg: Dict[str, Any],
) -> RuleOutput:
    # Get true rule
    true_rule = env.rule

    # Load test dataset:
    test_trajectories = env.unwrapped.get_test_dataset()

    all_dict: Dict[str, Any] = {
        "rules": [],
        "weights": [],
        "counts": [],
        "importance_probs": [],
        "likelihoods": [],
        "test_likelihoods": [],
    }
    for _ in tqdm(range(cfg["nb_collecting"]), desc="Importance Sampling iterations"):
        # Generate trajectories
        prompt_trajectories = generate_text_trajectories(
            env, agent, true_rule, cfg["nb_trajectories"]
        )
        # Sample rules
        rules, importance_probs = generate_rules(
            theorist, prompt_trajectories, cfg["nb_rules"]
        )
        # Get unique rules and counts
        rules, counts, importance_probs = get_unique_rules(rules, importance_probs)
        if cfg["add_true_rule"]:
            rules.append(true_rule.get_prompt())
            counts = np.append(counts, 1)
            importance_probs = np.append(
                importance_probs, np.log(1 / (cfg["nb_rules"] + 1))
            )
        if cfg["add_without_rule"]:
            rules.append(None)
            counts = np.append(counts, 1)
            importance_probs = np.append(
                importance_probs, np.log(1 / (cfg["nb_rules"] + 1))
            )
        # Compute likelihoods of new data using the rules
        likelihoods = compute_likelihood(statistician, rules, prompt_trajectories)

        # weights is just the likelihoods for importance sampling with resampling
        weights = np.log(counts) + likelihoods - importance_probs

        all_dict["rules"].extend(rules)
        all_dict["weights"].extend(weights)
        all_dict["counts"].extend(counts)
        all_dict["importance_probs"].extend(importance_probs)
        all_dict["likelihoods"].extend(likelihoods)

    # Compute likelihoods of test data for the rules
    all_dict["test_likelihoods"] = compute_likelihood(
        statistician, all_dict["rules"], test_trajectories
    )
    # Print rules and weights sorted
    indices = np.argsort(-np.array(all_dict["test_likelihoods"]))
    print("------------------------")
    print("true rule: " + repr(true_rule))
    print("------------------------")
    for ind in indices:
        print(
            f"-----rule-----:   {repr(all_dict['rules'][ind])}\n weight: {all_dict['weights'][ind]:2f}, importance: {all_dict['importance_probs'][ind]:2f}, likelihood: {all_dict['likelihoods'][ind]:2f}, count: {all_dict['counts'][ind]}, test_likelihood: {all_dict['test_likelihoods'][ind]:2f}"
        )
    return RuleOutput(true_rule, all_dict["rules"], all_dict["likelihoods"], all_dict)

from copy import copy
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from utils.utils_env import BaseAgent, Trajectory, generate_text_trajectories
from utils.utils_llm import (
    LlmModel,
    compute_likelihood,
    evolve_rules,
    generate_rules,
    score_rules,
)
from utils.utils_save import RuleOutput
from worldllm_envs.base import BaseRuleEnv


def get_worst_trajectories(
    logp: np.ndarray, trajectories: List[Trajectory], num_worst_trajectories: int
):
    """Return the worst trajectories according to the log probabilities"""
    arr_worst_ind = np.argsort(logp, axis=1)[:, :num_worst_trajectories]
    return [
        [trajectories[incr_worst_ind] for incr_worst_ind in worst_ind]
        for worst_ind in arr_worst_ind
    ]


def metropolis_hastings(
    env: BaseRuleEnv,
    agent: BaseAgent,
    theorist: LlmModel,
    statistician: LlmModel,
    cfg: Dict[str, Any],
    curriculum_rules: List[str],
) -> RuleOutput:
    """Metropolis-Hasting algorithm"""

    # Load test dataset:
    test_trajectories = env.unwrapped.get_test_dataset()

    all_dict: Dict[str, Any] = {
        "rules": [],
        "current_true_rule": [],
        "weights": [],
        "importance_probs": [],
        "likelihoods": [],
        "test_likelihoods": [],
    }

    # Generate trajectories
    rule_to_test = curriculum_rules[0]
    prompt_trajectories = generate_text_trajectories(
        env, agent, rule_to_test, cfg["nb_trajectories"]
    )
    # Sample rules
    if cfg["first_rules"] is not None:
        prev_rules = cfg["first_rules"]
        assert len(prev_rules) == cfg["nb_rules"]
    else:
        prev_rules, _ = generate_rules(theorist, prompt_trajectories, cfg["nb_rules"])
    if cfg["num_worst_trajectories"] and cfg["num_worst_trajectories"] > 0:
        prev_likelihoods, all_logp = compute_likelihood(
            statistician, prev_rules, prompt_trajectories, return_all_logp=True
        )
        prev_worst_trajectories = get_worst_trajectories(
            all_logp, prompt_trajectories, cfg["num_worst_trajectories"]
        )
        all_worst_trajectories = copy(prev_worst_trajectories)
    else:
        prev_likelihoods = compute_likelihood(
            statistician, prev_rules, prompt_trajectories
        )
    all_dict["rules"] = copy(prev_rules)
    all_dict["likelihoods"] = prev_likelihoods
    all_dict["importance_probs"] = [0] * cfg["nb_rules"]
    all_dict["prev_rules_ind"] = [-1] * cfg["nb_rules"]
    all_dict["weights"] = [0] * cfg["nb_rules"]
    all_dict["current_true_rule"] = [
        curriculum_rules[0] for _ in range(cfg["nb_rules"])
    ]
    all_dict["nb_rules"] = cfg["nb_rules"]
    prev_rules_ind = np.zeros((cfg["nb_rules"],), dtype=int)
    for incr_collecting in tqdm(
        range(cfg["nb_collecting"]), desc="Collecting iterations"
    ):
        for i in tqdm(
            range(cfg["nb_iterations"]),
            desc="Metropolis-Hastings iterations",
            leave=False,
        ):
            if (
                cfg["num_worst_trajectories"] is not None
                and cfg["num_worst_trajectories"] > 0
            ):
                # Sample a new rule
                rules, importance_probs = evolve_rules(
                    theorist,
                    prompt_trajectories,
                    prev_rules,
                    worst_trajectories=prev_worst_trajectories,
                )
                # Compute likelihoods of new data using the rules
                likelihoods, all_logp = compute_likelihood(
                    statistician, rules, prompt_trajectories, return_all_logp=True
                )
                worst_trajectories = get_worst_trajectories(
                    all_logp, prompt_trajectories, cfg["num_worst_trajectories"]
                )
                if cfg["use_hasting_correction"]:
                    # Compute reverse kernel
                    rev_importance_probs = score_rules(
                        theorist,
                        prompt_trajectories,
                        prev_rules,
                        rules,
                        worst_trajectories=worst_trajectories,
                    )
            else:
                # Sample a new rule
                rules, importance_probs = evolve_rules(
                    theorist, prompt_trajectories, prev_rules
                )
                # Compute likelihoods of new data using the rules
                likelihoods = compute_likelihood(
                    statistician, rules, prompt_trajectories
                )
                if cfg["use_hasting_correction"]:
                    # Compute reverse kernel
                    rev_importance_probs = score_rules(
                        theorist, prompt_trajectories, prev_rules, rules
                    )

            if cfg["use_hasting_correction"]:
                # Compute weights
                weights = (
                    likelihoods
                    - prev_likelihoods
                    - importance_probs
                    + rev_importance_probs
                )
            else:
                # Compute weights
                weights = likelihoods - prev_likelihoods
            # Update rules obtained
            all_dict["rules"].extend(rules)
            all_dict["weights"].extend(weights)
            all_dict["likelihoods"] = np.append(all_dict["likelihoods"], likelihoods)
            all_dict["prev_rules_ind"].extend(prev_rules_ind)
            all_dict["current_true_rule"].extend(
                [rule_to_test for _ in range(len(rules))]
            )
            # Accept or reject
            mask = np.where(np.log(np.random.rand()) < weights, 1, 0)
            prev_rules_ind = np.where(
                mask, (incr_collecting * cfg["nb_iterations"]) + i + 1, prev_rules_ind
            )
            prev_rules = np.where(mask, rules, prev_rules)
            prev_likelihoods = np.where(mask, likelihoods, prev_likelihoods)
            if (
                cfg["num_worst_trajectories"] is not None
                and cfg["num_worst_trajectories"] > 0
            ):
                all_worst_trajectories.extend(worst_trajectories)
                prev_worst_trajectories = np.where(
                    np.tile(mask, (len(worst_trajectories[0]), 1)).T,
                    worst_trajectories,
                    prev_worst_trajectories,
                )
        # Regenerate trajectories if not last iteration
        if incr_collecting + 1 < cfg["nb_collecting"]:
            # Regenerate trajectories
            rule_to_test = curriculum_rules[
                ((incr_collecting + 1) * len(curriculum_rules)) // cfg["nb_collecting"]
            ]
            prompt_trajectories = generate_text_trajectories(
                env, agent, rule_to_test, cfg["nb_trajectories"]
            )
            # Recompute the likelihoods for the new trajectories
            prev_likelihoods, all_logp = compute_likelihood(
                statistician, prev_rules, prompt_trajectories, return_all_logp=True
            )
            prev_worst_trajectories = get_worst_trajectories(
                all_logp, prompt_trajectories, cfg["num_worst_trajectories"]
            )
    # Compute likelihoods of test data for the rules
    all_dict["test_likelihoods"] = compute_likelihood(
        statistician, all_dict["rules"], test_trajectories
    )
    indices = np.argsort(-np.array(all_dict["test_likelihoods"]))
    for ind in indices:
        print(
            f"-----true_rule-----: {all_dict['current_true_rule'][ind]}, rule:  {ind%len(rules)}-{ind//len(rules)}({all_dict['prev_rules_ind'][ind]}):   {repr(all_dict['rules'][ind])}\nlikelihood: {all_dict['likelihoods'][ind]:2f}, weight: {all_dict['weights'][ind]:2f}, test_likelihood: {all_dict['test_likelihoods'][ind]:2f}"
        )
    return RuleOutput(
        curriculum_rules,
        all_dict["rules"],
        all_dict["likelihoods"],
        all_dict,
    )

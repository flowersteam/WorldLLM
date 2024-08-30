from copy import copy
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

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
) -> RuleOutput:
    """Metropolis-Hasting algorithm"""
    # Get true rule
    true_rule = env.rule

    # Generate trajectories
    prompt_trajectories = generate_text_trajectories(
        env, agent, true_rule, cfg["nb_trajectories"]
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
    all_rules = copy(prev_rules)
    all_likelihoods = prev_likelihoods
    all_prev_rules_ind = [-1] * cfg["nb_rules"]
    all_weights = [0] * cfg["nb_rules"]
    prev_rules_ind = np.zeros((cfg["nb_rules"],), dtype=int)
    for i in tqdm(range(cfg["nb_iterations"]), "Metropolis-Hastings iterations"):
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
            likelihoods = compute_likelihood(statistician, rules, prompt_trajectories)
            # Compute reverse kernel
            rev_importance_probs = score_rules(
                theorist, prompt_trajectories, prev_rules, rules
            )
        weights = (
            likelihoods - prev_likelihoods - importance_probs + rev_importance_probs
        )
        # Update rules obtained
        all_rules.extend(rules)
        all_weights.extend(weights)
        all_likelihoods = np.append(all_likelihoods, likelihoods)
        all_prev_rules_ind.extend(prev_rules_ind)
        # Accept or reject
        mask = np.where(np.log(np.random.rand()) < weights, 1, 0)
        prev_rules_ind = np.where(mask, i + 1, prev_rules_ind)
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
    indices = np.argsort(-np.array(all_likelihoods))
    print("------------------------")
    print("true rule: " + repr(true_rule))
    print("------------------------")
    for ind in indices:
        print(
            f"-----rule-----:{ind%len(rules)}-{ind//len(rules)}({all_prev_rules_ind[ind]}):   {repr(all_rules[ind])}, likelihood: {all_likelihoods[ind]:2f}, weight: {all_weights[ind]:2f}"
        )
    return RuleOutput(
        true_rule,
        all_rules,
        all_likelihoods,
        {
            "weights": all_weights,
            "prev_rules_ind": all_prev_rules_ind,
            "nb_particles": cfg["nb_rules"],
        },
    )

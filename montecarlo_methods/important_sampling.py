import os
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from utils.utils_llm import LlmModel, Statistician, compute_likelihood, generate_rules
from utils.utils_save import RuleOutput
from utils.utils_sb3 import SB3Agent
from worldllm_envs.base import BaseAgent, BaseRuleEnv


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
    experimenter: BaseAgent,
    theorist: LlmModel,
    statistician: Statistician,
    cfg: Dict[str, Any],
    output_dir: str,
) -> RuleOutput:
    """Important sampling algorithm for rule discovery. Return logs and rules."""
    # Load test dataset:
    test_trajectories = env.unwrapped.test_dataset

    all_dict: Dict[str, Any] = {
        "rules": [],
        "current_true_rule": [],
        "best_rule": [],
        "weights": [],
        "importance_probs": [],
        "likelihoods": [],
    }
    best_rule = None
    prev_best_rule = None
    best_rule_likelihood: float
    # region Main loop
    for i in tqdm(range(cfg["nb_phases"]), desc="Collecting iterations"):
        reset_info = {
            "pipeline_progression": i / cfg["nb_phases"],
            "stat_rule": best_rule,
            "env_rule": env.unwrapped.rule,
        }
        # 1. Generate trajectories
        prompt_trajectories, set_discovered_transitions = (
            experimenter.generate_trajectories(
                env,
                cfg["nb_trajectories"],
                reset_info,
                (
                    experimenter.model.n_steps * experimenter.model.n_envs
                    if isinstance(experimenter, SB3Agent)
                    else None
                ),
            )
        )
        # Take smaller subset to generate the rules
        subset_trajectories = prompt_trajectories[-cfg["nb_subset_traj"] :]
        # Update seen transitions for the statistician
        statistician.prompt_info.discovered_transitions.update(
            set_discovered_transitions
        )
        # Recompute likelihood for the best rule
        best_rule_likelihood = compute_likelihood(
            statistician,
            [best_rule],
            subset_trajectories,
        )[0][0]
        # 2. Generate rules
        rules, importance_probs = generate_rules(
            theorist, subset_trajectories, cfg["nb_rules"] * cfg["nb_iterations"]
        )
        # 3. Compute likelihoods of new data using the rules
        likelihoods, _ = compute_likelihood(statistician, rules, subset_trajectories)

        # 4.weights is just the likelihoods for importance sampling with resampling
        weights = likelihoods - importance_probs

        # 5. Logs
        all_dict["rules"].extend(rules)
        all_dict["weights"].extend(weights)
        all_dict["importance_probs"].extend(importance_probs)
        all_dict["likelihoods"].extend(likelihoods)
        all_dict["current_true_rule"].extend(
            [env.unwrapped.get_rule() for _ in range(len(rules))]
        )
        # Change best rule if different than the previous one
        best_rule_ind = np.argmax(likelihoods)
        if best_rule_likelihood < likelihoods[best_rule_ind] or prev_best_rule is None:
            prev_best_rule = best_rule
            best_rule = rules[best_rule_ind]

        all_dict["best_rule"].append(best_rule)

        if isinstance(experimenter, SB3Agent):
            # Prepare the rewards and score the experimenter
            new_rewards = experimenter.compute_reward(
                cfg,
                statistician,
                best_rule,
                prev_best_rule,
                prompt_trajectories,
            )
            # Train the experimenter
            experimenter.train_step(new_rewards)

        if i > 0 and i % cfg["save_every"] == 0:
            output = RuleOutput(
                all_dict["rules"],
                all_dict["likelihoods"],
                all_dict,
            )
            # Save output
            output.to_json(os.path.join(output_dir, f"all_{i}.json"))
            # Save experimenter if sb3
            if isinstance(experimenter, SB3Agent):
                experimenter.model.save(os.path.join(output_dir, f"experimenter_{i}"))
    # endregion
    # Add all transtion to the statistician for scoring the test
    statistician.prompt_info.discovered_transitions = env.get_all_transition_to_prompt()
    # Compute likelihoods of test data for the rules
    all_dict["test_likelihoods_best"], all_dict["test_transition_scores_best"] = (
        compute_likelihood(statistician, all_dict["best_rule"], test_trajectories)
    )
    print("Importance Sampling done")
    return RuleOutput(all_dict["rules"], all_dict["likelihoods"], all_dict)

"""Main file for the Metropolis-Hastings algorithm"""

import os
from copy import copy
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from tqdm import tqdm

from utils.utils_llm import (
    LlmModel,
    Statistician,
    compute_likelihood,
    evolve_rules,
    generate_rules,
    score_rules,
)
from utils.utils_save import RuleOutput
from utils.utils_sb3 import SB3Agent
from worldllm_envs.base import BaseAgent, BaseRuleEnv, Trajectory


def get_worst_trajectories(
    logp: np.ndarray, trajectories: List[Trajectory], num_worst_trajectories: int
) -> List[List[Trajectory]]:
    """Return the worst trajectories according to the log probabilities

    Args:
        logp (np.ndarray): Log probabilities of all the  trajectories.
        trajectories (List[Trajectory]): List of trajectories.
        num_worst_trajectories (int): Number of the worst trajectories to return.

    Returns:
        List[List[Trajectory]]: List of lists containing the worst trajectories.
    """
    arr_worst_ind = np.argsort(logp, axis=1)[:, :num_worst_trajectories]
    return [
        [trajectories[incr_worst_ind] for incr_worst_ind in worst_ind]
        for worst_ind in arr_worst_ind
    ]


def metropolis_hastings(
    env: BaseRuleEnv,
    experimenter: BaseAgent,
    theorist: LlmModel,
    statistician: Statistician,
    cfg: Dict[str, Any],
    output_dir: str,
) -> RuleOutput:
    """Metropolis-Hasting algorithm, return logs to save"""

    # Load test dataset:
    test_trajectories = env.unwrapped.test_dataset

    all_dict: Dict[str, Any] = {
        "rules": [],
        "current_true_rule": [],
        "best_rule": [],
        "best_rule_ind": [],
        "weights": [],
        "importance_probs": [],
        "likelihoods": [],
        "nb_subset_transitions": [],
        "nb_all_transitions": [],
        "transitions": [],
    }
    add_worst_trajectories = cfg["num_worst_trajectories"] > 0
    # region Initialize and first loop of the algorithm
    # 1. Generate trajectories

    reset_info = {
        "pipeline_progression": 0,
        "stat_rule": None,
        "env_rule": env.unwrapped.get_rule(),
    }
    prompt_trajectories, set_discovered_transitions, lst_transitions = (
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
    # Update seen transitions for the statistician
    statistician.prompt_info.discovered_transitions.update(set_discovered_transitions)
    # Take smaller subset to generate the first rules
    subset_trajectories = prompt_trajectories[-cfg["nb_subset_traj"] :]
    subset_lst_transitions = lst_transitions[-cfg["nb_subset_traj"] :]
    # Add trajectories to log
    all_dict["transitions"].append(prompt_trajectories)
    # Log the transitions
    unique_transi = np.unique(
        [transi for sublist in lst_transitions for transi in sublist],
        return_counts=True,
    )
    unique_subset_transi = np.unique(
        [transi for sublist in subset_lst_transitions for transi in sublist],
        return_counts=True,
    )

    all_dict["nb_all_transitions"].append(
        {key: value for key, value in zip(*unique_transi)}
    )
    all_dict["nb_subset_transitions"].append(
        {key: value for key, value in zip(*unique_subset_transi)}
    )
    # 2. Sample rules
    if cfg["first_rules"] is not None:
        prev_rules = cfg["first_rules"]
        assert len(prev_rules) == cfg["nb_rules"]
    else:
        prev_rules, _ = generate_rules(theorist, subset_trajectories, cfg["nb_rules"])
    # 3. Score rules
    if add_worst_trajectories:
        (prev_likelihoods, all_logp), _ = compute_likelihood(
            statistician, prev_rules, subset_trajectories, return_all_logp=True
        )
        prev_worst_trajectories = get_worst_trajectories(
            all_logp, subset_trajectories, cfg["num_worst_trajectories"]
        )
        all_worst_trajectories = copy(prev_worst_trajectories)
    else:
        prev_likelihoods, _ = compute_likelihood(
            statistician, prev_rules, subset_trajectories
        )

    # 4. Get best rule
    best_rule_ind = np.argmax(prev_likelihoods)
    best_rule = prev_rules[best_rule_ind]
    prev_best_rule = None
    # 5. Log the rules
    all_dict["rules"] = copy(prev_rules)
    all_dict["likelihoods"] = prev_likelihoods
    all_dict["importance_probs"] = [0] * cfg["nb_rules"]
    all_dict["prev_rules_ind"] = [-1] * cfg["nb_rules"]
    all_dict["weights"] = [0] * cfg["nb_rules"]
    all_dict["current_true_rule"] = [
        env.unwrapped.get_rule() for _ in range(cfg["nb_rules"])
    ]
    all_dict["nb_rules"] = cfg["nb_rules"]
    all_dict["best_rule"] = [best_rule]
    all_dict["best_rule_ind"] = [best_rule_ind]
    prev_rules_ind = np.zeros((cfg["nb_rules"],), dtype=int)
    # endregion

    # region Main loop of the algorithm
    for i in tqdm(
        range(cfg["nb_phases"]),
        desc="Loop iterations",
    ):
        # 1. Regenerate trajectories
        reset_info = {
            "pipeline_progression": i / cfg["nb_phases"],
            "stat_rule": best_rule,
            "env_rule": env.unwrapped.get_rule(),
        }
        prompt_trajectories, set_discovered_transitions, lst_transitions = (
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
        subset_lst_transitions = lst_transitions[-cfg["nb_subset_traj"] :]
        # Add trajectories to log
        all_dict["transitions"].append(prompt_trajectories)
        # Update seen transitions for the statistician
        statistician.prompt_info.discovered_transitions.update(
            set_discovered_transitions
        )
        # Log the transitions
        unique_transi = np.unique(
            [transi for sublist in lst_transitions for transi in sublist],
            return_counts=True,
        )
        unique_subset_transi = np.unique(
            [transi for sublist in subset_lst_transitions for transi in sublist],
            return_counts=True,
        )

        all_dict["nb_all_transitions"].append(
            {key: value for key, value in zip(*unique_transi)}
        )
        all_dict["nb_subset_transitions"].append(
            {key: value for key, value in zip(*unique_subset_transi)}
        )
        # Recompute the likelihoods for the new trajectories
        (prev_likelihoods, all_logp), _ = compute_likelihood(
            statistician, prev_rules, subset_trajectories, return_all_logp=True
        )
        prev_worst_trajectories = get_worst_trajectories(
            all_logp, subset_trajectories, cfg["num_worst_trajectories"]
        )
        # Do Multiple Metropolis Hastings steps
        for incr_mh in tqdm(
            range(cfg["nb_iterations"]), desc="Metropolis-Hastings", leave=False
        ):
            # Metropolis-Hastings step
            if add_worst_trajectories:
                # Sample a new rule
                rules, importance_probs = evolve_rules(
                    theorist,
                    subset_trajectories,
                    prev_rules,
                    worst_trajectories=prev_worst_trajectories,
                )
                # Compute likelihoods of new data using the rules
                (likelihoods, all_logp), _ = compute_likelihood(
                    statistician, rules, subset_trajectories, return_all_logp=True
                )
                worst_trajectories = get_worst_trajectories(
                    all_logp, subset_trajectories, cfg["num_worst_trajectories"]
                )
            else:
                # Sample a new rule
                rules, importance_probs = evolve_rules(
                    theorist, subset_trajectories, prev_rules
                )
                # Compute likelihoods of new data using the rules
                likelihoods, _ = compute_likelihood(
                    statistician, rules, subset_trajectories
                )
            if cfg["use_hasting_correction"]:
                # Compute reverse kernel
                rev_importance_probs = score_rules(
                    theorist,
                    subset_trajectories,
                    prev_rules,
                    rules,
                    worst_trajectories=(
                        worst_trajectories if add_worst_trajectories else None
                    ),
                )
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
                [env.unwrapped.get_rule() for _ in range(len(rules))]
            )
            # Accept or reject
            mask = np.where(np.log(np.random.rand()) < weights, 1, 0)
            prev_rules_ind = np.where(
                mask,
                (i * cfg["nb_iterations"]) + incr_mh + 1,
                prev_rules_ind,
            )
            prev_rules = np.where(mask, rules, prev_rules)
            prev_likelihoods = np.where(mask, likelihoods, prev_likelihoods)
            if add_worst_trajectories:
                all_worst_trajectories.extend(worst_trajectories)
                prev_worst_trajectories = np.where(
                    np.tile(mask, (len(worst_trajectories[0]), 1)).T,
                    worst_trajectories,
                    prev_worst_trajectories,
                )
        # Change best rule if different than the previous one
        best_rule_ind = np.argmax(prev_likelihoods)
        if prev_rules[best_rule_ind] != best_rule or prev_best_rule is None:
            prev_best_rule = best_rule
            best_rule = prev_rules[best_rule_ind]

        all_dict["best_rule"].append(best_rule)
        all_dict["best_rule_ind"].append(best_rule_ind)

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
    print("Metropolis Hastings done")
    return RuleOutput(
        all_dict["rules"],
        all_dict["likelihoods"],
        all_dict,
    )

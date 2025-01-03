import argparse
import json
import os
import sys

import gymnasium

# Add the project root directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.utils_llm import LlmAgent, LlmModel, build_exp_prompt_info, load_transformers
from worldllm_envs.base import BaseWrapper, Trajectory

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index", type=int, required=True, help="Index of the script to run"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Seed for the environment"
    )
    args = parser.parse_args()

    env: BaseWrapper = gymnasium.make(
        "worldllm_envs/PlaygroundText-v1",
        **{
            "max_steps": 30,
            "seed": args.seed,
            "playground_config": {"max_nb_objects": 8},
        },
    )
    config = {
        "name": "microsoft/Phi-3-mini-4k-instruct",
        "model_params": {},
        "tokenizer_params": {},
        "is_quantized": True,
        "generation_kwargs": {"cache_implementation": None, "cache_config": None},
        "chat_template": "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'system') %}{{'<|system|>' + '\n' + message['content'] + '<|end|>' + '\n'}}{% elif (message['role'] == 'user') %}{{'<|user|>' + '\n' + message['content'] + '<|end|>' + '\n' + '<|assistant|>' + '\n'}}{% elif message['role'] == 'assistant' %}{{message['content'] + '<|end|>' + '\n'}}{% endif %}{% endfor %}",
    }
    experimenter_model = load_transformers(config)
    exp_prompt_info = build_exp_prompt_info(
        experimenter_model,
        env.unwrapped.get_message_info(),
        "You like doing a lot of puzzles. Please answer with a brief answer and be as precise as you can.",
        5,
    )
    experimenter = LlmModel(
        experimenter_model[0],
        experimenter_model[1],
        exp_prompt_info,
    )
    agent = LlmAgent(env.action_space)
    agent.set_model(experimenter)
    goal_to_test = {
        "grow any plant": "The goal is to grow any plant.",
        "grow any small_herbivorous": "The goal is to grow any small herbivorous.",
        "grow any big_herbivorous": "The goal is to grow any big herbivorous.",
        "grow any small_herbivorous then grow any big_herbivorous": "The goal is to grow any small herbivorous then grow any big herbivorous.",
    }
    lst_stat_rule = [
        None,
        "1. Water does not change when interacted with.\n2. Only the object being interacted with can change its kind when interacted with.\n3. Vegetable seeds become their corresponding vegetables when watered.\n4. Identical vegetables, when combined, transform into a single vegetable.\n5. Holding identical baby animals results in a single object of that animal'0;0000024e18000000\"\na: you go to the water\na: you are standing on the water\na: you pick up the object\na: you are holding the water\na: you go to the potato seed\na: you are standing on the potato seed\na: you give the water\na: the water and potato seed transform into the potato\na: you pick up the object\na: you are holding the potato\na: you go to",
        "1. Always interact with water first: Move to water (`a: You go to the water. o: You are standing on the water.`).\n2. Collect the object: Pick it up after you are on or near water (`a: You pick up the object. o: You are holding the water.`).\n3. Encounter transformations: Seeds transform into their respective fruits/vegetables; only then do they interact with water. Note the type of object is crucial for transformation (`a: You give the water. o: The object and water transform into the object.`).\n4. If no water, stay put and interact when in proximity or on water.\n5. When all objects are on water, pick those up; if you do nothing, no interaction occurs (`a: You go to the [object]. o: If on water, you may pick it up, otherwise, genuin0`¦",
        "To grow a plant, you need to grab water and release it on the seeds on the plant. To grow a small herbivorous, you need to grab a plant and release it on the baby small herbivorous. To grow a big herbivorous, you need to grab 2 plants and release them on the baby big herbivorous. You can only hold 2 objects at a time.",
    ]
    success_rate = []
    time_to_reach_goal = []
    lst_trajectories = []
    n_episodes = 50
    lst_goals = list(goal_to_test.keys())
    goal = lst_goals[args.index % len(lst_goals)]
    stat_rule = lst_stat_rule[args.index // len(lst_goals)]
    for _ in range(n_episodes):
        print("\nCurrent rule:", stat_rule, "Current_goal:", goal)
        new_rule = goal
        obs, info = env.reset(options={"rule": new_rule})
        info["stat_rule"] = stat_rule
        info["env_rule"] = goal
        # Compute plan
        agent.reset(info)
        done = False
        n_steps = 0
        while not done:
            # Record inputs from keyboard
            # Print possible actions
            action, _ = agent(obs, **info)
            obs, reward, done, _, info = env.step(action)
            n_steps += 1
        # If goal is reached
        if info["success"]:
            success_rate.append(1)
        else:
            success_rate.append(0)
        lst_trajectories.append(
            Trajectory(
                info["trajectory_obs_text"],
                info["trajectory_act_text"],
                info["trajectory_diff_text"],
            )
        )
        time_to_reach_goal.append(n_steps)
    # Save results
    if not os.path.exists("eval_results"):
        os.makedirs("eval_results")
    with open(f"eval_results/results_{args.index}.json", "w") as f:
        json.dump(
            {
                "success_rate": success_rate,
                "time_to_reach_goal": time_to_reach_goal,
                "stat_rule": stat_rule,
                "goal": goal,
                "trajectories": [
                    trajectory.to_dict() for trajectory in lst_trajectories
                ],
            },
            f,
        )

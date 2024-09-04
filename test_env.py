import gymnasium
from tqdm import tqdm

from worldllm_envs.base import BaseRuleEnv
from worldllm_envs.playground.playground_text_wrapper import PerfectAgent

success_rate = 0
for seed in tqdm(range(10000)):
    try:
        env: BaseRuleEnv = gymnasium.make(
            "worldllm_envs/PlaygroundText-v1", **{"max_steps": 20, "seed": seed}
        )
        new_rule = env.unwrapped.generate_rule()
        obs, info = env.reset(options={"rule": new_rule})
        agent = PerfectAgent(env.action_space, info["obj_dict"], info["goal"])
        index = 0

        done = False
        while not done:
            # Record inputs from keyboard
            action = agent(obs)
            obs, _, done, _, info = env.step(action)
            index += 1

        # print("\n".join(info["text_trajectory"]))
        assert agent.is_done == True
        success_rate += 1
    except Exception as e:
        print(e)
        print(seed)
print(success_rate / 10000)

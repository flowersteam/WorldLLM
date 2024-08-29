from gymnasium.envs.registration import register

from worldllm_envs.base import TextWrapper

register(
    id="worldllm_envs/Door-v0",
    entry_point="worldllm_envs.door.door:DoorEnv",
    additional_wrappers=(TextWrapper.wrapper_spec(),),
)

register(
    id="worldllm_envs/PlaygroundText-v1",
    entry_point="worldllm_envs.playground.playground_text_wrapper:PlayGroundText",
    max_episode_steps=50,
)

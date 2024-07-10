from gymnasium.envs.registration import register

from worldllm_envs.envs.base import TextWrapper

register(
    id="worldllm_envs/Door-v0",
    entry_point="worldllm_envs.envs.door:DoorEnv",
    additional_wrappers=(TextWrapper.wrapper_spec(),),
)

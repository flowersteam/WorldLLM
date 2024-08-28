from gymnasium.envs.registration import register

from worldllm_envs.base import TextWrapper

register(
    id="worldllm_envs/Door-v0",
    entry_point="worldllm_envs.door.door:DoorEnv",
    additional_wrappers=(TextWrapper.wrapper_spec(),),
)

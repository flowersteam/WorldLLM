from gymnasium.envs.registration import register

register(
    id="worldllm_envs/Door-v0",
    entry_point="worldllm_envs.envs.door:DoorEnv",
)

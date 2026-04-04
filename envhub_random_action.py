import torch
from lerobot.envs.factory import make_env

# Load from the hub
envs_dict = make_env("LightwheelAI/leisaac_env:envs/so101_pick_orange.py", n_envs=1, trust_remote_code=True)

# Access the environment
suite_name = next(iter(envs_dict))
sync_vector_env = envs_dict[suite_name][0]
# retrieve the isaac environment from the sync vector env
env = sync_vector_env.envs[0].unwrapped

# Use it like any gym environment
obs, info = env.reset()

while True:
    action = torch.tensor(env.action_space.sample())
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()

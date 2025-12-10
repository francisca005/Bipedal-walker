from gymnasium.envs.registration import register
from .injured_bipedal import InjuredBipedalWalker

register(
    id="InjuredBipedalWalker-v0",
    entry_point="my_envs.injured_bipedal:InjuredBipedalWalker",
    max_episode_steps=1600,
)
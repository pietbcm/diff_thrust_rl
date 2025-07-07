import sys
sys.path.append("./jsbgym")
import jsbgym
import gymnasium as gym

from gymnasium.envs import registry

print([env.id for env in registry.values() if "C172" in env.id])

env = gym.make("C172-HeadingControlTask-Shaping.STANDARD-FG-v0", render_mode="flightgear")
env.reset()
env.render()
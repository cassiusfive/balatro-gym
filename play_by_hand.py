import balatro_gym # noqa
import gymnasium as gym

from gymnasium.wrappers import FlattenObservation

env = gym.make("BalatroSmall-v0", render_mode="ansi")
env = FlattenObservation(env)
observation = env.reset()

done = False
while not done:
    print(env.render())

    print(env.get_wrapper_attr('valid_actions')())
    action = int(input("Enter action: "))
    obs, reward, done, truncated, info = env.step(action)
    print(reward)

env.close()
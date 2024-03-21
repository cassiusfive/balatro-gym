import balatro_gym # noqa
import gymnasium as gym

env = gym.make("Balatro-v0", render_mode="ansi")

observation = env.reset()

done = False
while not done:
    print(env.render())

    print(env.get_wrapper_attr('valid_actions')())
    action = int(input("Enter action: "))
    env.step(action)

env.close()
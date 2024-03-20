import balatro_gym
import gymnasium as gym

env = gym.make("Balatro-v0", render_mode="ansi")

observation = env.reset()

done = False
while not done:
    print(env.render())

    action = int(input("Enter action: "))
    env.step(action)

env.close()
import gym
import numpy as np

env = gym.make(id="CustomCartPoleContinuous-v1")
# env = gym.make(id="CartPole-v0")
observation = env.reset()

cnt = 0
all_reward = 0
avr_reward = 0

while cnt < 10:

    # randomly pick an action
    # action = env.action_space.sample()

    x1 = observation[0]
    x2 = observation[1]
    x3 = observation[2]
    x4 = observation[3]

    # expression: x4 < 0.0(Mul(exp(0.1),Pow(log(0.1),-1)),x4)
    # temp = np.log(0.1)
    # temp = np.exp(0.1) / temp
    # action = temp if x4 < 0.0 else x4

    # expression: x3 < 0.21(x3,log(Mul(5.0,Pow(Mul(exp(cos(log(x2))),cos(exp(x3))),-1))))
    # temp = np.exp(np.cos(np.log(x2)))
    # temp = np.cos(np.exp(x3)) * temp
    # temp = 5.0 / temp
    # temp = np.log(temp)
    # action = x3 if x3 < 0.21 else temp

    # expression: x2 < 0.0(Mul(x3,Pow(exp(exp(Mul(exp(x1),x2))),-1)),x3)
    # temp = np.exp(np.exp(np.exp(x1) * x2))
    # temp = x3 / temp
    # action = temp if x2 < 0.0 else x3

    # expression: x₃ + sin(0.1⋅x₁)
    # action = x3 + np.sin(0.1 * x1)

    # expression: sin(x₃⋅log(x₄ + 1.6094379124341))
    # action = np.sin(x3 * np.log(x4 + 1.6094379124341))

    # expression: x₃⋅cos(exp(x₁⋅x₄⋅(1.10517091807565 - x₂)))
    # action = x3 * np.cos(np.exp(x1 * x4 * (1.10517091807565 - x2)))

    # expression: x3 < 0.21(x3,exp(sin(Mul(x1,Pow(Mul(5.0,Pow(exp(exp(0.1)),-1)),-1)))))
    # temp = np.exp(np.exp(0.1))
    # temp = 5.0 / temp
    # temp = x1 / temp
    # temp = np.sin(temp)
    # temp = np.exp(temp)
    # action = x3 if x3 < 0.21 else temp

    # expression: x1 < 2.4(x3,Mul(0.1,Pow(sin(exp(x3)),-1)))
    # temp = np.sin(np.exp(x3))
    # temp = 0.1 / temp
    # action = x3 if x1 < 2.4 else temp

    # expression: x1 < 2.4(x3,sin(Mul(0.1,log(Add(log(5.0),log(Mul(Mul(Add(1.0,x3),x1),x3)))))))
    temp = np.log((1 + x3) * x1 * x3)
    temp = np.log(np.log(5.0) + temp)
    temp = np.sin(0.1 * temp)
    action = x3 if x1 < 2.4 else temp
    
    # action = 0 if action < 0 else 1
    action = np.clip(action, env.action_space.low, env.action_space.high)
    observation, reward, done, info = env.step(action)
    env.render()
    all_reward += reward
    if done:
        cnt += 1
        print("Reward: ", all_reward)
        avr_reward += all_reward
        all_reward = 0
        observation = env.reset()

print("Average reward of 10 episodes: ", avr_reward / 10)
env.close()
import gym
import numpy as np
import cv2 as cv
from dso.vision_module.vision_observer import VisionObserver

env = gym.make('CustomCartPoleContinuous-v1')
# env = gym.make('CartPole-v0')
env.reset()
start = True
cnt = 0
all_reward = 0
avr_reward = 0
vision_observer = None

while cnt < 10:
    frame = env.render(mode='rgb_array')
    if start:
        vision_observer = VisionObserver()
        observation = vision_observer.start(frame)
        start = False
    else:
        observation = vision_observer.step(frame)
    x = observation

    # expression: x4 < 0.0(Mul(exp(0.1),Pow(log(0.1),-1)),x4)
    # temp = np.log(0.1)
    # temp = np.exp(0.1) / temp
    # action = temp if x[3] < 0.0 else x[3]

    # expression: 0.0909090909090909⋅cos(x₄)/(x₄ + 0.109964966682941⋅exp(cos(0.2⋅x₄)))
    # temp = np.exp(np.cos(0.2 * x[3]))
    # temp = x[3] + 0.109964966682941 * temp
    # action = 0.0909090909090909 * np.cos(x[3]) / temp

    # expression: -x10+x20+1.0/(x16+x3-(x7*sin(x10-x21)/x2))
    temp = x[6] * np.sin(x[9] - x[20]) / x[1]
    temp = x[15] + x[2] - temp
    action = -x[9] + x[19] + 1.0 / temp

    # expression: x20/(x27 - cos(sin(sin(cos(x14⋅x8⋅(x26 + 5.0) + x18)))))
    # action = x[19]/(x[26] - np.cos(np.sin(np.sin(np.cos(x[13]*x[7]*(x[25] + 5.0) + x[17])))))

    # expression: x15*x4*(-x12 + x18)*exp(cos(x13 + x16 - x9 - cos(x14 - x4) + cos(log(x8))))
    # action = x[14]*x[3]*(x[17]-x[11])*np.exp(np.cos(x[12]+x[16]-x[8]-np.cos(x[13]-x[3])+np.cos(np.log(x[7]))))

    # expression: sin(x2)/(-0.1*x10 + 0.1*x12 - 0.1*x17 + x18 + 0.1*x18*exp(-sin(x17 + x3 - cos(x2))))
    # action = np.sin(x[1]) / (-0.1*x[9] + 0.1*x[11] - 0.1*x[16] + x[17] + 0.1*x[17]*np.exp(-1*np.sin(x[16] + x[2] - np.cos(x[1]))))


    # expression: 0.1 - x18 / (x1+x5+sin(cos(log(x1-x1*exp(-x6*exp(0.1/x16)))))+0.1)
    # action = 0.1 - x[17] / (x[0]+x[4]+np.sin(np.cos(np.log(x[0]-x[0]*np.exp(-1*x[5]*np.exp(0.1/x[15])))))+0.1)

    # expression: x18 - x6 - sin(x3/(x16-x18*(x5+sin(x1/cos(x16/log(0.1/(148.413159102577*x1+x7+5.0)))))))
    # action = x[17] - x[5] - np.sin(x[2]/(x[15]-x[17]*(x[4]+np.sin(x[0]/np.cos(x[15]/np.log(0.1/(148.413159102577*x[0]+x[6]+5.0)))))))

    # expression: 148.413159102577*exp(-x5+cos(0.5*x11/x4-x17*sin(sin(sin(x13-cos(x4)+cos(log(x12)*sin(exp(exp(x16)*cos(x12-exp(x16-x3+x8)))))))))+x3*exp(x2))
    # action = 148.413159102577*np.exp(-1*x[4]+np.cos(0.5*x[10]/x[3]-x[16]*np.sin(np.sin(np.sin(x[12]-np.cos(x[3])+np.cos(np.log(x[11])*np.sin(np.exp(np.exp(x[15])*np.cos(x[11]-np.exp(x[15]-x[2]+x[7])))))))))+x[2]*np.exp(x[1]))

    action = np.clip(action, env.action_space.low, env.action_space.high)
    observation, reward, done, info = env.step(action)
    if not vision_observer.render():
        break
    all_reward += reward

    if done:
        env.reset()
        cnt += 1
        start = True
        print('Reward: ', all_reward)
        avr_reward += all_reward
        all_reward = 0

print('Average reward: ', avr_reward / 10)
cv.destroyAllWindows()
env.close()
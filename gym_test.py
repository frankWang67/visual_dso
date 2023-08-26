import gym
import numpy as np
import cv2 as cv

env = gym.make(id='CartPole-v0')
observation = env.reset()

for i in range(3):
    img = env.render(mode='rgb_array')
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    cv.imwrite('images/test/frame_{}.png'.format(i), img)
    observation, reward, done, info = env.step(action=1)


# img1 = cv.imread('images/test/img1.png')
# img2 = cv.imread('images/test/img2.png')
# cv.imshow('img1', img1)
# cv.imshow('img2', img2)
# if cv.waitKey(0) == 32:
#     cv.destroyAllWindows()
env.close()
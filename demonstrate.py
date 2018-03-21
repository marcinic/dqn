

import gym
import keras
import numpy as np
import atari_wrappers
from dqn import DQN
from atari_wrappers import make_atari,wrap_deepmind

"""
Load previously trained network and produce a video of gameplay

"""


env = gym.make('BreakoutNoFrameskip-v4')
env = atari_wrappers.WarpFrame(env)
env = atari_wrappers.FrameStack(env,4)

state = env.reset()
moves = env.action_space.n
#env = wrappers.Monitor(env,'/home/marcinic/hdd/projects/reinforcement/dqn/experiment-1',force=True)

agent = DQN((84,84),moves)
agent.load_model("model/dqn.h5")
num_episodes = 10
num_frames = 1000
for e in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        env.render()
        action = np.argmax(agent.model.predict(np.reshape(state,(1,84,84,4))))#env.action_space.sample()
        state,reward,done,info = env.step(action)

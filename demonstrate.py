

import gym
import keras
import numpy as np

"""
Load previously trained network and produce a video of gameplay

"""


env = gym.make('BreakoutDeterministic-v4')
env.reset()
moves = env.action_space.n
env = wrappers.Monitor(env,'/home/marcinic/hdd/projects/reinforcement/dqn/experiment-1',force=True)


nn = agent.load_model("model/dqn.h5")
